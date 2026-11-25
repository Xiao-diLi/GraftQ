"""GraftQ multimodal GNN-Transformer architecture."""

import torch
import torch.nn as nn

from ..layers.decoder import TransformerDecoder
from ..layers.encoder import GnnTransformerEncoder


class GNNTransformer(nn.Module):
    """Graph-Augmented Fusion Transformer for Quantum Circuit Optimization (GraftQ).

    Jointly encodes a simplified ZX diagram (via GCN) and source gate tokens in a
    Pre-LN Transformer encoder with RoPE, then autoregressively decodes an optimized
    gate sequence with a Pre-LN Transformer decoder.
    """

    def __init__(
        self,
        d_model: int,
        gnn_in_features: int,
        vocab_size: int,
        nhead: int,
        dim_feedforward: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        num_gnn_layers: int,
        dropout: float = 0.0,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()

        self.encoder = GnnTransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_encoder_layers,
            gnn_in_features=gnn_in_features,
            num_gnn_layers=num_gnn_layers,
            num_heads=nhead,
            d_ff=dim_feedforward,
            dropout=dropout,
            gradient_checkpointing=gradient_checkpointing,
        )

        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_decoder_layers,
            num_heads=nhead,
            d_ff=dim_feedforward,
            dropout=dropout,
            gradient_checkpointing=gradient_checkpointing,
        )

    def _create_pad_mask(self, seqs: torch.Tensor, pad_idx: int) -> torch.Tensor:
        return (seqs != pad_idx).unsqueeze(1)

    def _create_subsequent_mask(self, size: int, device: torch.device) -> torch.Tensor:
        return (torch.triu(torch.ones(size, size, device=device), diagonal=1) == 0).unsqueeze(0)

    def decode(self, src: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        tgt_sub_mask = self._create_subsequent_mask(tgt.size(1), tgt.device)
        tgt_mask = tgt_sub_mask.unsqueeze(1)
        logits, _ = self.decoder(
            tgt=tgt,
            memory=src,
            tgt_mask=tgt_mask,
            memory_mask=src_mask,
        )
        return logits

    def forward(self, batch: dict) -> torch.Tensor:
        src, src_mask = self.encoder(batch["src"], batch["src_len"], batch["graph"])
        logits = self.decode(src, src_mask, batch["tgt"][:, :-1])
        return logits
