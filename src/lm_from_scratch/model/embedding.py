import torch
from torch import nn


class Embedding(nn.Module):
    """Look up token vectors: (...) integer IDs become (..., embedding_dim)."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        if num_embeddings <= 0 or embedding_dim <= 0:
            raise ValueError("Embedding dimensions must be positive.")

        scaled_weight = torch.nn.init.trunc_normal_(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype),
            mean=0,
            std=1,
            a=-3,
            b=3,
        )

        self.weight = nn.Parameter(scaled_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x]
