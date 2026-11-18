import torch
from torch import nn
from lm_from_scratch.model.linear import Linear
from lm_from_scratch.model.positional import RotaryPositionalEmbedding
from lm_from_scratch.model.scaled_dot_product_attention import scaled_dot_product_attention

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: int | None = None, max_seq_len: int | None = None) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        if theta is None and max_seq_len is None:
            self.rope = None
        elif theta is None or max_seq_len is None:
            raise ValueError(
                "Provide both theta and max_seq_len to enable RoPE."
            )
        else:
            self.rope = RotaryPositionalEmbedding(
                theta=theta,
                d_k=self.d_head,
                max_seq_len=max_seq_len,
            )

        self.q_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.o_proj = Linear(d_model, d_model)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        *batch_dims, seq_len, d_model = x.shape
        
        head_shape = (
            *batch_dims,
            seq_len, 
            self.num_heads,
            self.d_head
        )
        
        # Q, K, V: (B, n_heads, seq_len, d_head)
        Q = self.q_proj(x).reshape(head_shape).transpose(-3, -2)
        K = self.k_proj(x).reshape(head_shape).transpose(-3, -2)
        V = self.v_proj(x).reshape(head_shape).transpose(-3, -2)
        
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        
        causal_masking_tensor = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        attention_output = scaled_dot_product_attention(Q, K, V, causal_masking_tensor) # (B, n_heads, seq_len, d_head)
        attention_output = attention_output.transpose(-3, -2).reshape(*batch_dims, seq_len, d_model) # (B, seq_len, d_head)
        
        attention_output = self.o_proj(attention_output)
        
        return attention_output
