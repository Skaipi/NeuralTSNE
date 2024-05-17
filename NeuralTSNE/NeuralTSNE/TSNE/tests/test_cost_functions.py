import pytest
import torch
from NeuralTSNE.TSNE import CostFunctions

from NeuralTSNE.TSNE.tests.fixtures.parametric_tsne_fixtures import (
    default_parametric_tsne_instance,
)


@pytest.mark.parametrize(
    "P, Q, expected",
    [
        (
            torch.tensor([[0.1, 0.4, 0.5], [0.3, 0.25, 0.45], [0.26, 0.24, 0.5]]),
            torch.tensor([[0.8, 0.15, 0.05], [0.1, 0.5, 0.4], [0.3, 0.4, 0.4]]),
            torch.tensor(29.623),
        ),
        (
            torch.tensor(
                [
                    [0.0000, 0.0592, 0.0651, 0.0372, 0.0588],
                    [0.0592, 0.0000, 0.0528, 0.0382, 0.0533],
                    [0.0651, 0.0528, 0.0000, 0.0444, 0.0465],
                    [0.0372, 0.0382, 0.0444, 0.0000, 0.0446],
                    [0.0588, 0.0533, 0.0465, 0.0446, 0.0000],
                ]
            ),
            torch.tensor(
                [
                    [0.4781, 0.7788],
                    [0.8525, 0.3280],
                    [0.0730, 0.9723],
                    [0.0679, 0.1797],
                    [0.5947, 0.5116],
                ]
            ),
            torch.tensor(0.0154),
        ),
    ],
)
def test_kl_divergence(
    default_parametric_tsne_instance,
    P: torch.tensor,
    Q: torch.tensor,
    expected: torch.tensor,
):
    tsne_instance, _ = default_parametric_tsne_instance
    tsne_instance.batch_size = P.shape[0]
    C = CostFunctions.kl_divergence(Q, P, {"device": "cpu", "batch_size": P.shape[0]})

    assert torch.allclose(C, expected, rtol=1e-3)
