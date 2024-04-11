from unittest.mock import patch

import pytest

from NeuralTSNE.TSNE.parametric_tsne import ParametricTSNE


@pytest.fixture
def parametric_tsne_instance(request):
    params = request.param
    with (
        patch(
            "NeuralTSNE.TSNE.parametric_tsne.ParametricTSNE.set_loss_fn"
        ) as mock_loss_fn,
        patch("torchinfo.summary") as mock_summary,
        patch(
            "NeuralTSNE.TSNE.parametric_tsne.NeuralNetwork", autospec=True
        ) as mock_nn,
    ):
        mock_loss_fn.return_value = params["loss_fn"]
        instance = ParametricTSNE(**params)
        yield instance, params, {
            "loss_fn": mock_loss_fn,
            "summary": mock_summary,
            "nn": mock_nn,
        }


@pytest.fixture
def default_parametric_tsne_instance():
    params = {
        "loss_fn": "kl_divergence",
        "n_components": 2,
        "perplexity": 50,
        "batch_size": 10,
        "early_exaggeration_epochs": 10,
        "early_exaggeration_value": 8.0,
        "max_iterations": 500,
        "features": 15,  # TODO: Add the ability to inject crucial params as in Classifier class
        "multipliers": [1.0, 1.5],
        "n_jobs": 6,
        "tolerance": 1e-6,
        "force_cpu": True,
    }
    with patch("torchinfo.summary"):
        yield ParametricTSNE(**params), params
