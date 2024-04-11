from typing import Any

import pytest

from NeuralTSNE.TSNE.neural_network import NeuralNetwork


@pytest.fixture
def neural_network_params(
    request: type[pytest.FixtureRequest],
) -> dict[str, Any]:
    initial_features, n_components, multipliers, *args = request.param
    args_keys = ["pre_filled_layers"]

    return {
        "initial_features": initial_features,
        "n_components": n_components,
        "multipliers": multipliers,
    } | dict(zip(args_keys, args))


@pytest.fixture
def neural_network(neural_network_params: dict[str, Any]) -> NeuralNetwork:
    return NeuralNetwork(**neural_network_params)
