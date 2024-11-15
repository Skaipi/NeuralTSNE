from unittest.mock import patch

import pytest

from NeuralTSNE.TSNE import DimensionalityReduction


@pytest.fixture
def classifier_instance(request, default_parametric_tsne_instance):
    params = request.param

    with patch(
        "NeuralTSNE.TSNE.Modules.dimensionality_reduction.DimensionalityReduction.reset_exaggeration_status"
    ) as mock_exaggeration_status:
        yield DimensionalityReduction(
            tsne=default_parametric_tsne_instance[0], **params
        ), params | {
            "tsne": default_parametric_tsne_instance[0]
        }, mock_exaggeration_status


@pytest.fixture(params=[None])
def default_classifier_instance(request, default_parametric_tsne_instance):
    tsne_params = request.param or {}
    params = {"shuffle": True, "optimizer": "rmsprop", "lr": 1e-6}
    tsne_instance, default_tsne_params = default_parametric_tsne_instance
    for k, v in tsne_params.items():
        tsne_instance.__dict__[k] = v
    return DimensionalityReduction(tsne=tsne_instance, **params), {
        "tsne_params": tsne_params,
        "default_tsne_params": default_tsne_params,
        "params": params,
    }
