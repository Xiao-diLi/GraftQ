"""Utility functions for multimodal sequence concatenation and sampling masks."""

import torch
import torch.nn.functional as F


def concat_valid_sequences(
    A: torch.Tensor,
    B: torch.Tensor,
    L1: torch.Tensor,
    L2: torch.Tensor,
):
    """Pack valid elements of two padded sequences [A; B] with right-padding.

    Args:
        A: Tensor of shape [batch, len1, dim] (graph node embeddings).
        B: Tensor of shape [batch, len2, dim] (gate token embeddings).
        L1: Tensor of shape [batch] with valid lengths in A.
        L2: Tensor of shape [batch] with valid lengths in B.

    Returns:
        C: Concatenated tensor of shape [batch, max(L1) + max(L2), dim].
        valid_mask: Boolean mask of shape [batch, max(L1) + max(L2)].
    """
    _, len1, dim = A.shape
    device = A.device

    out_len = L1.max() + L2.max()
    AB = torch.cat([A, B], dim=1)

    grid = torch.arange(out_len, device=device).unsqueeze(0)
    L1 = L1.unsqueeze(1)
    L2 = L2.unsqueeze(1)

    mask_A = grid < L1
    shift = (len1 - L1) * (~mask_A).long()
    gather_indices = grid + shift

    valid_mask = grid < (L1 + L2)
    safe_indices = gather_indices.masked_fill(~valid_mask, 0)

    safe_indices_expanded = safe_indices.unsqueeze(-1).expand(-1, -1, dim)
    C = AB.gather(1, safe_indices_expanded)
    C = C * valid_mask.unsqueeze(-1).type_as(C)
    return C, valid_mask


def mask_with_top_k(logits: torch.Tensor, top_k: int = 0) -> torch.Tensor:
    """Filter logits using top-k truncation."""
    if top_k <= 0:
        return logits
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits[indices_to_remove] = -float("Inf")
    return logits


def mask_with_top_p(logits: torch.Tensor, top_p: float = 1.0) -> torch.Tensor:
    """Filter logits using nucleus (top-p) sampling."""
    if top_p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = -float("Inf")
    return logits
