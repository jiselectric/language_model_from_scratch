import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Normalize along dim after shifting finite logits to avoid overflow."""
    shifted = x - x.max(dim=dim, keepdim=True).values
    exp_values = shifted.exp()
    return exp_values / exp_values.sum(dim=dim, keepdim=True)
