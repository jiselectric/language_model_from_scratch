import pytest
import torch

from lm_from_scratch.model.attention import MultiHeadSelfAttention
from lm_from_scratch.model.linear import Linear
from lm_from_scratch.model.positional import RotaryPositionalEmbedding
from lm_from_scratch.model.transformer import TransformerLM


def test_model_without_rotary_embeddings():
    model = TransformerLM(32, 8, 16, 1, 2, 32)
    assert model(torch.zeros(2, 4, dtype=torch.long)).shape == (2, 4, 32)


@pytest.mark.parametrize("d_model,num_heads", [(16, 0), (15, 2), (0, 2)])
def test_attention_rejects_invalid_dimensions(d_model, num_heads):
    with pytest.raises(ValueError):
        MultiHeadSelfAttention(d_model, num_heads)


@pytest.mark.parametrize("width", [0, -2, 3])
def test_rotary_rejects_invalid_width(width):
    with pytest.raises(ValueError):
        RotaryPositionalEmbedding(10000.0, width, 8)


def test_linear_rejects_zero_dimensions():
    with pytest.raises(ValueError):
        Linear(0, 4)


def test_model_rejects_context_overflow():
    model = TransformerLM(32, 4, 16, 1, 2, 32, 10000.0)
    with pytest.raises(ValueError, match="context"):
        model(torch.zeros(2, 5, dtype=torch.long))
