from .softmax import softmax
import torch
import math

def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, masking: torch.Tensor | None = None):
    *_, query_embd_dim = query.shape
    *_, key_embd_dim = key.shape
    assert query_embd_dim == key_embd_dim
    
    x = query @ key.transpose(-2, -1) / math.sqrt(query_embd_dim)
    
    if masking is not None:
        x =  x.masked_fill(~masking, float('-inf'))
    
    return softmax(x, -1) @ value
