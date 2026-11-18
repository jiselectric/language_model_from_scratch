import torch
from torch import nn

from lm_from_scratch.model.embedding import Embedding
from lm_from_scratch.model.linear import Linear
from lm_from_scratch.model.normalization import RMSNorm
from lm_from_scratch.model.transformer_block import TransformerBlock


class TransformerLM(nn.Module):
    """Map (..., sequence) token IDs to (..., sequence, vocab_size) logits."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float | None = None,
    ) -> None:
        super().__init__()

        if context_length <= 0 or num_layers <= 0:
            raise ValueError("context_length and num_layers must be positive.")
        self.context_length = context_length

        self.embedding = Embedding(vocab_size, d_model)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model, num_heads, d_ff, rope_theta, context_length if rope_theta is not None else None
                )
                for _ in range(num_layers)
            ]
        )
        self.rms_norm = RMSNorm(d_model=d_model)
        self.linear = Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 1 or x.shape[-1] > self.context_length:
            raise ValueError("Token sequence must fit within the configured context length.")
        x = self.embedding(x)

        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        x = self.rms_norm(x)
        x = self.linear(x)
        return x
