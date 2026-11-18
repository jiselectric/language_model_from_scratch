import torch
from torch import nn

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
                
        variance = 2 / (in_features + out_features)
        std = variance**0.5
        a = -3 * std
        b = 3 * std
        
        scaled_weight = torch.nn.init.trunc_normal_(
            torch.empty(out_features, in_features, device=device, dtype=dtype), 
            mean=0, 
            std=std,
            a=a,
            b=b,
        )
        
        self.weight = nn.Parameter(scaled_weight)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T
