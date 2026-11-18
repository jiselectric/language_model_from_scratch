import torch
from torch import nn
from lm_from_scratch.model.embedding import Embedding
from lm_from_scratch.model.transformer_block import TransformerBlock
from lm_from_scratch.model.normalization import RMSNorm
from lm_from_scratch.model.linear import Linear

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: int | None = None):
        super().__init__()
        
        self.embedding = Embedding(vocab_size, d_model)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, rope_theta, context_length) for _ in range(num_layers)])
        self.rms_norm = RMSNorm(d_model=d_model)
        self.linear = Linear(d_model, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x)
        
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        x = self.rms_norm(x)
        x = self.linear(x)
        return x
