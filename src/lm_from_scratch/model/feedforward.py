import torch
from torch import nn

from lm_from_scratch.model.linear import Linear


class SwiGLU(nn.Module):
    """Gated feedforward layer preserving (..., d_model) input shape."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w1 = Linear(d_model, d_ff)  # gate weight matrix
        self.w2 = Linear(d_ff, d_model)  # output projection
        self.w3 = Linear(d_model, d_ff)  # value weight matrix

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.w1(x)
        gate = z * torch.sigmoid(z)
        value = self.w3(x)
        result = self.w2(gate * value)
        return result
