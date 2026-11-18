import numpy
import torch
import torch.nn.functional as F

from .adapters import run_softmax


def test_softmax_matches_pytorch():
    x = torch.tensor(
        [
            [0.4655, 0.8303, 0.9608, 0.9656, 0.6840],
            [0.2583, 0.2198, 0.9334, 0.2995, 0.1722],
            [0.1573, 0.6860, 0.1327, 0.7284, 0.6811],
        ]
    )
    expected = F.softmax(x, dim=-1)
    numpy.testing.assert_allclose(run_softmax(x, dim=-1).detach().numpy(), expected.detach().numpy(), atol=1e-5)
    # Test that softmax handles numerical overflow issues
    numpy.testing.assert_allclose(
        run_softmax(x + 100, dim=-1).detach().numpy(),
        expected.detach().numpy(),
        atol=1e-5,
    )
