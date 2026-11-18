import torch

def softmax(x: torch.Tensor, i: int) -> torch.tensor:
    shifted = x - x.max(dim=i, keepdim=True).values
    exp_values = shifted.exp()
    return exp_values / exp_values.sum(dim=i, keepdim=True)
