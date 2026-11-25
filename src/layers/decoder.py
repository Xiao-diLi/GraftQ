"""Pre-LN Transformer decoder with RoPE causal self-attention and KV caching."""

import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .layers import FFN, MultiHeadAttention


class DecoderLayer(nn.Module):
    """Pre-LN Transformer decoder layer."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout, is_cross_attn=True)
        self.feed_forward = FFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory_mask: torch.Tensor,
        self_attn_cache: dict = None,
        cross_kv: tuple = None,
    ):
        tgt_norm = self.norm1(tgt)
        attn_output, new_self_attn_cache = self.self_attn(
            tgt_norm, tgt_norm, tgt_norm, tgt_mask, cache=self_attn_cache
        )
        tgt = tgt + self.dropout1(attn_output)

        tgt_norm = self.norm2(tgt)
        attn_out, _ = self.cross_attn(
            tgt_norm, memory, memory, memory_mask, precomputed_kv=cross_kv
        )
        tgt = tgt + self.dropout2(attn_out)

        tgt = tgt + self.dropout3(self.feed_forward(self.norm3(tgt)))
        return tgt, new_self_attn_cache


class TransformerDecoder(nn.Module):
    """Autoregressive Pre-LN Transformer decoder with untied output projection."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
        gradient_checkpointing: bool,
    ):
        super().__init__()
        self.d_model = d_model
        self.gradient_checkpointing = gradient_checkpointing

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.embed_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def precompute_cross_kv(self, memory: torch.Tensor):
        """Precompute cross-attention key/value projections once per encoder memory."""
        cross_kv_cache = []
        bsz, mem_len, _ = memory.shape
        for layer in self.layers:
            num_heads = layer.cross_attn.num_heads
            d_k = layer.cross_attn.d_k
            k = layer.cross_attn.fc_k(memory).view(bsz, mem_len, num_heads, d_k).transpose(1, 2)
            v = layer.cross_attn.fc_v(memory).view(bsz, mem_len, num_heads, d_k).transpose(1, 2)
            cross_kv_cache.append((k, v))
        return cross_kv_cache

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory_mask: torch.Tensor,
        cache: list = None,
        cross_kv_cache: list = None,
    ):
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.embed_dropout(tgt_emb)

        new_cache = []
        x = tgt_emb
        for i, layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(fn):
                    def custom_forward(*inputs):
                        output, _ = fn(inputs[0], inputs[1], inputs[2], inputs[3], None)
                        return output
                    return custom_forward

                x = checkpoint(
                    create_custom_forward(layer),
                    x,
                    memory,
                    tgt_mask,
                    memory_mask,
                    use_reentrant=False,
                )
            else:
                self_attn_cache = cache[i] if cache is not None else None
                cross_kv = cross_kv_cache[i] if cross_kv_cache is not None else None
                x, kv_cache = layer(
                    x,
                    memory,
                    tgt_mask,
                    memory_mask,
                    self_attn_cache=self_attn_cache,
                    cross_kv=cross_kv,
                )
                new_cache.append(kv_cache)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits, new_cache
