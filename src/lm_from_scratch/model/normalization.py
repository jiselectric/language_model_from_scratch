import torch
from torch import nn


class RMSNorm(nn.Module):
    """Normalize the final feature dimension, accumulating squares in float32."""

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model <= 0 or eps <= 0:
            raise ValueError("d_model and eps must be positive.")
        self.eps = eps
        gains = torch.ones(d_model, device=device, dtype=dtype)
        self.gains = nn.Parameter(gains)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        mean_square = x.square().mean(-1, keepdim=True)
        rms = torch.sqrt(mean_square + self.eps)
        result = (x / rms) * self.gains

        return result.to(dtype=in_dtype)
