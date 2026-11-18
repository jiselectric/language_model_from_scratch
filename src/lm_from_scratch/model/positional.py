import torch
import torch.nn as nn


class RotaryPositionalEmbedding(nn.Module):
    """Rotate adjacent feature pairs using cached position-dependent angles."""

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        if d_k <= 0 or d_k % 2:
            raise ValueError("d_k must be positive and even.")
        if theta <= 0 or max_seq_len <= 0:
            raise ValueError("theta and max_seq_len must be positive.")

        # One frequency per adjacent feature pair.
        indices = torch.arange(
            0,
            d_k,
            2,
            device=device,
            dtype=torch.float32,
        )

        inv_freq = theta ** (-indices / d_k)

        # Cache positions up to the configured context length.
        positions = torch.arange(
            max_seq_len,
            device=device,
            dtype=torch.float32,
        )

        # Shape: (max_seq_len, d_k // 2)
        angles = positions[:, None] * inv_freq[None, :]

        # Nonpersistent buffers move with the module but are absent from checkpoints.
        self.register_buffer(
            "cos_cached",
            torch.cos(angles),
            persistent=False,
        )

        self.register_buffer(
            "sin_cached",
            torch.sin(angles),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:

        # Select rotations for the given positions.
        positions = token_positions.to(
            device=self.cos_cached.device,
            dtype=torch.long,
        )

        cos = self.cos_cached[positions].to(dtype=x.dtype)
        sin = self.sin_cached[positions].to(dtype=x.dtype)

        # Support shared token positions across extra
        # leading batch/head dimensions.
        while cos.ndim < x.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        # Split the last dimension into adjacent pairs.
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        # Apply the 2D rotation formula.
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        # Interleave the rotated coordinates.
        return torch.stack(
            (rotated_even, rotated_odd),
            dim=-1,
        ).flatten(-2)
