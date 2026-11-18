import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        gains = torch.ones(d_model, device=device, dtype=dtype)
        self.gains = nn.Parameter(
            gains
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        
        mean_square = x.square().mean(-1, keepdim=True) 
        rms = torch.sqrt(mean_square + self.eps)
        result = (x / rms) * self.gains
        
        return result.to(dtype=in_dtype)
