from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from NeuralTSNE.TSNE.helpers import Hbeta, x2p, x2p_job


@pytest.mark.parametrize(
    "D, beta, expected_H, expected_P",
    [
        (
            torch.tensor([[0.0, 2.0], [2.0, 0.0]]),
            0.5,
            1.2754,
            torch.tensor([[0.3655, 0.1345], [0.1345, 0.3655]]),
        ),
        (
            torch.tensor([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]]),
            0.7,
            1.9558,
            torch.tensor(
                [
                    [0.2114, 0.1050, 0.0521],
                    [0.1050, 0.2114, 0.0259],
                    [0.0521, 0.0259, 0.2114],
                ]
            ),
        ),
    ],
)
def test_Hbeta(D, beta, expected_H, expected_P):
    H, P = Hbeta(D, beta)
    assert torch.isclose(H, torch.tensor(expected_H), rtol=1e-3)
    assert torch.allclose(P, expected_P, rtol=1e-3)


@pytest.mark.parametrize("i", [7, 1, 21])
@pytest.mark.parametrize("perplexity", [10, 50, 1000])
@pytest.mark.parametrize("tolerance", [0.1, 0.01, 0.001, 1e-6])
@pytest.mark.parametrize("max_iterations", [100, 200, 50])
@pytest.mark.parametrize(
    "D",
    [
        torch.tensor([17.0, 89.0, 123.0, 40.0, 67.0]),
        torch.tensor([0.6, 0.2311, 0.456, 0.01, 1.53]),
    ],
)  # TODO: CHECK IF THIS IS CORRECT
def test_x2p_job(
    i: int,
    perplexity: int,
    D: torch.Tensor,
    tolerance: float,
    max_iterations: int,
):
    logU = torch.tensor([np.log(perplexity)], dtype=torch.float32)
    data = i, D, logU
    result = x2p_job(data, tolerance, max_iterations)
    i, P, Hdiff, iterations = result
    assert i == i
    if iterations != max_iterations:
        assert torch.allclose(Hdiff, torch.zeros_like(Hdiff), rtol=tolerance)
    else:
        estimated_tolerance = 10 ** torch.ceil(torch.log10(torch.abs(Hdiff)))[0]
        assert torch.isclose(torch.zeros_like(Hdiff), Hdiff, atol=estimated_tolerance)


@pytest.mark.parametrize(
    "X, perplexity, tolerance, expected_shape",
    [
        [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), 2, 0.1, (2, 2)],
        [
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
            3,
            0.01,
            (3, 3),
        ],
        [
            torch.tensor(
                [
                    [1.0, 6.0, 4.0, 2.0, 5.0],
                    [7.0, 2.0, 6.0, 7.0, 3.0],
                    [8.0, 2.0, 1.0, 7.0, 9.0],
                ]
            ),
            5,
            0.001,
            (3, 3),
        ],
    ],
)
@patch("NeuralTSNE.TSNE.helpers.x2p_job")
def test_x2p(
    mock_x2p_job: MagicMock,
    X: torch.Tensor,
    perplexity: int,
    tolerance: float,
    expected_shape: tuple,
):
    log_perplexity = torch.tensor([np.log(perplexity)], dtype=torch.float32)
    pair_squared_distances = torch.cdist(X, X, p=2) ** 2

    pairwise_distances = pair_squared_distances[pair_squared_distances != 0].reshape(
        pair_squared_distances.shape[0], -1
    )

    returned_P = [
        (i, torch.ones(pairwise_distances.shape[1])) for i in range(X.shape[0])
    ]

    mock_x2p_job.side_effect = returned_P

    P = x2p(X, perplexity, tolerance)
    assert P.shape == expected_shape
    assert mock_x2p_job.call_count == X.shape[0]
    mock_x2p_calls = mock_x2p_job.call_args_list

    P_diag = P.diag()
    assert torch.allclose(P_diag, torch.zeros_like(P_diag), rtol=1e-5)
    P_no_diag = P[P != 0].reshape(P.shape[0], -1)
    assert torch.allclose(P_no_diag, torch.ones_like(P_no_diag), rtol=1e-5)

    for k, call_args in enumerate(mock_x2p_calls):
        data, tol = call_args[0]
        i, D, logU = data

        assert i == k
        assert torch.allclose(D, pairwise_distances[k], rtol=1e-5)
        assert torch.isclose(logU, log_perplexity, rtol=1e-5)
        assert tol == tolerance
