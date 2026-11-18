import math

import torch

from .softmax import softmax


def scaled_dot_product_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, masking: torch.Tensor | None = None
) -> torch.Tensor:
    """Attend from (..., queries, d_k) to (..., keys, d_v); True means allowed."""
    *_, query_embd_dim = query.shape
    *_, key_embd_dim = key.shape
    if query_embd_dim <= 0 or query_embd_dim != key_embd_dim:
        raise ValueError("Query and key feature dimensions must match and be positive.")
    if key.shape[-2] != value.shape[-2]:
        raise ValueError("Key and value sequence lengths must match.")
    if masking is not None and masking.dtype != torch.bool:
        raise TypeError("Attention mask must have boolean dtype.")

    x = query @ key.transpose(-2, -1) / math.sqrt(query_embd_dim)

    if masking is not None:
        x = x.masked_fill(~masking, float("-inf"))

    return softmax(x, -1) @ value
