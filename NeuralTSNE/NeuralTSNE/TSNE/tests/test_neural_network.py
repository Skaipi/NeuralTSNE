import pytest
import torch
from collections import OrderedDict
from typing import List

from NeuralTSNE.TSNE.tests.fixtures.neural_network_fixtures import (
    neural_network_params,
    neural_network,
)


@pytest.mark.parametrize(
    "neural_network_params",
    [
        ((10, 5, [2, 3, 1, 0.5])),
        ((5, 3, [2, 6, 1])),
        ((3, 2, [2, 1])),
    ],
    indirect=True,
)
def test_neural_network_forward(neural_network_params, neural_network):
    input_data = torch.randn(10, neural_network_params["initial_features"])
    output = neural_network(input_data)

    assert output.shape == (10, neural_network_params["n_components"])


@pytest.mark.parametrize(
    "neural_network_params, activation_functions",
    [
        (
            (10, 2, [2, 3, 4, 2, 0.3, 1.4]),
            [(i, torch.nn.ReLU) for i in range(1, 13, 2)],
        ),
        ((20, 3, [2, 6, 1]), [(i, torch.nn.ReLU) for i in range(1, 7, 2)]),
        ((15, 4, [2, 1]), [(i, torch.nn.ReLU) for i in range(1, 5, 2)]),
    ],
    indirect=["neural_network_params"],
)
def test_neural_network_layer_shapes(
    neural_network_params, neural_network, activation_functions: List[torch.nn.Module]
):
    input_data = torch.randn(10, neural_network_params["initial_features"])
    layer_shapes = [input_data.shape[1]]
    is_activation_valid = []

    for i, layer in enumerate(neural_network.sequential_stack):
        if len(activation_functions) > 0 and i == activation_functions[0][0]:
            _, function = activation_functions.pop(0)
            if isinstance(layer, function):
                is_activation_valid.append(True)
            else:
                is_activation_valid.append(False)
        else:
            layer_shapes.append(layer.out_features)

    expected_shapes = [
        int(
            neural_network_params["multipliers"][i]
            * neural_network_params["initial_features"]
        )
        for i in range(len(neural_network_params["multipliers"]))
    ]

    expected_shapes = [layer_shapes[0]] + expected_shapes + [layer_shapes[-1]]
    assert expected_shapes == layer_shapes
    assert activation_functions == []
    assert all(is_activation_valid)


@pytest.mark.parametrize(
    "neural_network_params",
    [
        (
            (
                10,
                5,
                [2, 3, 1, 0.5],
                OrderedDict(
                    {
                        "0": torch.nn.Linear(10, 20),
                        "ReLu0": torch.nn.ReLU(),
                        "1": torch.nn.Linear(20, 30),
                        "GeLu1": torch.nn.GELU(),
                        "2": torch.nn.Linear(30, 5),
                        "ELu2": torch.nn.ELU(),
                        "3": torch.nn.Linear(5, 5),
                    }
                ),
            )
        )
    ],
    indirect=True,
)
def test_neural_network_pre_filled_layers(neural_network_params, neural_network):
    pre_filled_layers = neural_network_params["pre_filled_layers"]
    neural_network_pre_filled = torch.nn.Sequential(pre_filled_layers)

    for layer, pre_filled_layer in zip(
        neural_network.sequential_stack, neural_network_pre_filled
    ):
        assert layer == pre_filled_layer


@pytest.mark.parametrize(
    "neural_network_params", [((10, 5, [2, 3, 1, 0.5]))], indirect=True
)
def test_neural_network_gradients(neural_network_params, neural_network):
    input_data = torch.randn(10, neural_network_params["initial_features"])
    target = torch.rand(10, neural_network_params["n_components"])

    loss_fn = torch.nn.MSELoss(reduction="sum")
    optimizer = torch.optim.SGD(neural_network.parameters(), lr=1e-3)

    neural_network.train()

    y_pred = neural_network(input_data)
    loss = loss_fn(y_pred, target)
    optimizer.zero_grad()

    gradient = neural_network.sequential_stack[0].weight
    gradient = gradient.clone().detach()

    loss.backward()

    optimizer.step()

    gradient_after = neural_network.sequential_stack[0].weight

    assert not torch.allclose(gradient, gradient_after, atol=1e-8)
