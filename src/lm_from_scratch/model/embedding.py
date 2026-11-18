import torch
from torch import nn

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        
        scaled_weight = torch.nn.init.trunc_normal_(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype),
            mean=0,
            std=1,
            a=-3,
            b=3,
        )
        
        self.weight = nn.Parameter(scaled_weight)
    
    def forward(self, x):
        return self.weight[x]
