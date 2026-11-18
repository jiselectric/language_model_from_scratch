import torch
from torch import nn
from lm_from_scratch.model.normalization import RMSNorm
from lm_from_scratch.model.attention import MultiHeadSelfAttention
from lm_from_scratch.model.feedforward import SwiGLU

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: int | None = None, max_seq_len: int | None = None):
        super().__init__()
        
        self.rms_norm1 = RMSNorm(d_model=d_model)
        self.rms_norm2 = RMSNorm(d_model=d_model)
        self.multi_head_self_attention = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, theta=theta, max_seq_len=max_seq_len)
        self.ffnn = SwiGLU(d_model, d_ff)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first_sub_layer = x + self.multi_head_self_attention(self.rms_norm1(x))
        output = first_sub_layer + self.ffnn(self.rms_norm2(first_sub_layer))
        
        return output
