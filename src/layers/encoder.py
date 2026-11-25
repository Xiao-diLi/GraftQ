"""Multimodal GNN-Transformer encoder jointly processing ZX diagrams and gate tokens."""

import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_batch

from .layers import FFN, MultiHeadAttention
from .utils import concat_valid_sequences


class EncoderLayer(nn.Module):
    """Pre-LN Transformer encoder layer with RoPE self-attention."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        src_norm = self.norm1(src)
        attn_out, _ = self.self_attn(src_norm, src_norm, src_norm, src_mask)
        src = src + self.dropout1(attn_out)

        ffn_out = self.feed_forward(self.norm2(src))
        src = src + self.dropout2(ffn_out)
        return src


class GnnTransformerEncoder(nn.Module):
    """Encodes a ZX diagram with GCN layers, concatenates gate-token embeddings,
    and processes the joint sequence through a Pre-LN Transformer encoder.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        gnn_in_features: int,
        num_gnn_layers: int,
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

        self.gnn = nn.ModuleList(
            [GCNConv(gnn_in_features, d_model)]
            + [GCNConv(d_model, d_model) for _ in range(num_gnn_layers - 1)]
        )

        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.final_norm = nn.LayerNorm(d_model)

    def gnn_encode(self, graph):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        for conv in self.gnn:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(conv, x, edge_index, use_reentrant=False).relu()
            else:
                x = conv(x, edge_index).relu()

        node_repr, node_mask = to_dense_batch(x, batch)
        return node_repr, node_mask

    def forward(self, src: torch.Tensor, src_len: torch.Tensor, graph):
        graph_emb, graph_mask = self.gnn_encode(graph)
        graph_size = graph_mask.sum(-1).long()

        src_emb = self.embedding(src) * math.sqrt(self.d_model)

        x, mask = concat_valid_sequences(graph_emb, src_emb, graph_size, src_len)
        x = self.embed_dropout(x)
        mask = mask[:, None, None]

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(layer, x, mask, use_reentrant=False)
            else:
                x = layer(x, mask)

        x = self.final_norm(x)
        return x, mask
