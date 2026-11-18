import torch
from torch import nn

from lm_from_scratch.model.linear import Linear
from lm_from_scratch.model.positional import RotaryPositionalEmbedding
from lm_from_scratch.model.scaled_dot_product_attention import scaled_dot_product_attention


class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head attention with optional rotary position embeddings."""

    def __init__(
        self, d_model: int, num_heads: int, theta: float | None = None, max_seq_len: int | None = None
    ) -> None:
        super().__init__()
        if d_model <= 0 or num_heads <= 0 or d_model % num_heads:
            raise ValueError("d_model and num_heads must be positive; d_model must be divisible by num_heads.")

        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        if theta is None and max_seq_len is None:
            self.rope = None
        elif theta is None or max_seq_len is None:
            raise ValueError("Provide both theta and max_seq_len to enable RoPE.")
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

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        *batch_dims, seq_len, d_model = x.shape

        head_shape = (*batch_dims, seq_len, self.num_heads, self.d_head)

        # Queries, keys, values: (..., num_heads, sequence, d_head).
        query = self.q_proj(x).reshape(head_shape).transpose(-3, -2)
        key = self.k_proj(x).reshape(head_shape).transpose(-3, -2)
        value = self.v_proj(x).reshape(head_shape).transpose(-3, -2)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            query = self.rope(query, token_positions)
            key = self.rope(key, token_positions)

        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        attention_output = scaled_dot_product_attention(
            query, key, value, causal_mask
        )  # (..., num_heads, sequence, d_head)
        attention_output = attention_output.transpose(-3, -2).reshape(
            *batch_dims, seq_len, d_model
        )  # (..., sequence, d_model)

        attention_output = self.o_proj(attention_output)

        return attention_output
