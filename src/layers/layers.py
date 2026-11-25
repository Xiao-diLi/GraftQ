"""Transformer building blocks: FFN and Multi-Head Attention with RoPE and KV caching."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import RotaryPositionalEmbeddings, apply_rotary_pos_emb


class FFN(nn.Module):
    """Position-wise ReLU feed-forward network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention (with RoPE) or cross-attention (with optional precomputed KV)."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        is_cross_attn: bool = False,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.d_model = d_model
        self.is_cross_attn = is_cross_attn

        if not self.is_cross_attn:
            self.rope = RotaryPositionalEmbeddings(d_model=self.d_k)

        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)
        self.flash_attention_supported = hasattr(F, "scaled_dot_product_attention")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
        cache: dict = None,
        precomputed_kv: tuple = None,
    ):
        bsz, query_len, _ = query.shape
        q = self.fc_q(query).view(bsz, query_len, self.num_heads, self.d_k).transpose(1, 2)

        if precomputed_kv is not None:
            k, v = precomputed_kv
            new_cache = None
        else:
            _, key_len, _ = key.shape
            k = self.fc_k(key).view(bsz, key_len, self.num_heads, self.d_k).transpose(1, 2)
            v = self.fc_v(value).view(bsz, key_len, self.num_heads, self.d_k).transpose(1, 2)

            if not self.is_cross_attn:
                assert query_len == key_len
                offset = (
                    cache["len"]
                    if (cache is not None and "len" in cache)
                    else (cache["k"].shape[2] if cache is not None else 0)
                )
                cos, sin = self.rope(q, query_len, offset=offset)
                q, k = apply_rotary_pos_emb(q, k, cos, sin)

            if cache is not None:
                if "len" in cache:
                    step = cache["len"]
                    cache["k"][:, :, step : step + key_len] = k
                    cache["v"][:, :, step : step + key_len] = v
                    cache["len"] = step + key_len
                    k = cache["k"][:, :, : cache["len"]]
                    v = cache["v"][:, :, : cache["len"]]
                else:
                    prev_k, prev_v = cache["k"], cache["v"]
                    k = torch.cat([prev_k, k], dim=2)
                    v = torch.cat([prev_v, v], dim=2)

            if cache is not None and "len" in cache:
                new_cache = cache
            else:
                new_cache = {"k": k, "v": v}

        if self.flash_attention_supported:
            context = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False,
            )
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            context = torch.matmul(attn_weights, v)

        context = context.transpose(1, 2).contiguous().view(bsz, query_len, self.d_model)
        output = self.fc_o(context)
        return output, new_cache
