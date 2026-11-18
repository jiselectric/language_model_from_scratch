import torch
from torch import nn
from lm_from_scratch.model.linear import Linear

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        # d_ff = 64 * math.ceil((8 * d_model / 3) / 64) 
        self.w1 = Linear(d_model, d_ff) # gate weight matrix
        self.w2 = Linear(d_ff, d_model) # ffnn weight matrix
        self.w3 = Linear(d_model, d_ff) # value weight matrix
                
    def forward(self, x):
        z = self.w1(x) 
        gate = z * torch.sigmoid(z)
        value = self.w3(x)
        result = self.w2(gate * value)
        return result
