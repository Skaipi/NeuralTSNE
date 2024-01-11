from unittest.mock import MagicMock, call, patch, mock_open
import string
import tempfile
import io
import random
from collections import OrderedDict
from typing import List, Tuple, Any
import argparse
import os

import pytest
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import NeuralTSNE.TSNE as tsne


class PersistentStringIO(io.StringIO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._closed = False

    def close(self):
        self._closed = True

    @property
    def closed(self):
        return self._closed


class MyDataset(Dataset):
    def __init__(
        self,
        num_samples: int,
        num_variables: int,
        item_range: Tuple[float, float] | Tuple[int, int] = None,
        generate_int: bool = False,
    ):
        self.num_samples = num_samples
        self.num_variables = num_variables
        self.item_range = item_range or (0, 1)
        self.generate_int = generate_int

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        if self.generate_int:
            sample = torch.randint(*self.item_range, size=(self.num_variables,))
        else:
            sample = torch.FloatTensor(self.num_variables).uniform_(*self.item_range)
        return tuple([sample])


class DataLoaderMock:
    def __init__(self, dataset: MyDataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size
        self.batches = []

    def __iter__(self):
        for i in range(0, len(self.dataset), self.batch_size):
            batch = tuple(
                torch.cat(
                    [
                        torch.unsqueeze(self.dataset[j][k], 0)
                        for j in range(i, i + self.batch_size)
                    ],
                    dim=0,
                )
                for k in range(len(self.dataset[0]))
            )
            self.batches.append(batch)
            yield batch

    def __len__(self):
        return len(self.dataset)


def get_random_string(length: int):
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


@pytest.mark.parametrize(
    "a, b, to, epsilon, expected",
    [
        (1, 2, 3, 0.1, True),
        (1, 2, 4, 0.01, False),
        (1, 2.05, 3.06, 0.1, True),
        (1, 2.05, 3.06, 0.001, False),
    ],
)
def test_does_sum_up_to(a, b, to, epsilon, expected):
    assert tsne.does_sum_up_to(a, b, to, epsilon) == expected


@pytest.mark.parametrize(
    "input, expected",
    [
        ([1, 2, 10], [0.0, 1 / 9, 1.0]),
        ([[5, 6, 7], [9, 3, 6]], [[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]]),
        (
            [[2, 7, 9], [4, 3, 5], [8, 1, 6]],
            [[0.0, 1.0, 1.0], [2 / 6, 2 / 6, 0.0], [1.0, 0.0, 1 / 4]],
        ),
    ],
)
def test_normalize_columns(input, expected):
    tensor = torch.tensor(input)
    expected = torch.tensor(expected)
    result = tsne.normalize_columns(tensor)
    assert torch.allclose(result, expected)


@pytest.mark.parametrize(
    "data_variance, variance_threshold, filtered_data", [([5, 7, 10, 3, 4], 6, [1, 2])]
)
def test_filter_data_by_variance(data_variance, variance_threshold, filtered_data):
    sample_size = 10000

    artificial_data = [
        np.random.normal(0, variance**0.5, sample_size) for variance in data_variance
    ]
    artificial_data = torch.tensor(np.array(artificial_data).T)

    filtered_data = artificial_data[:, filtered_data]

    result = tsne.filter_data_by_variance(artificial_data, variance_threshold)
    assert torch.allclose(result, filtered_data)


@pytest.mark.parametrize(
    "data, filtered_data",
    [
        (
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            [[1.0, 2.0], [4.0, 5.0], [7.0, 8.0]],
        ),
        (
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            [[1.0, 3.0], [4.0, 6.0], [7.0, 9.0]],
        ),
        ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], None),
    ],
)
@patch("builtins.open", new_callable=mock_open)
def test_save_means_and_vars(
    mock_open: MagicMock,
    data: List[List[float]],
    filtered_data: List[List[float]] | None,
):
    file_handle = PersistentStringIO()
    mock_open.return_value = file_handle

    data_means = np.mean(data, axis=0)
    data_vars = np.var(data, axis=0, ddof=1)

    filtered_data_means = (
        None if filtered_data is None else np.mean(filtered_data, axis=0)
    )
    filtered_data_vars = (
        None if filtered_data is None else np.var(filtered_data, axis=0, ddof=1)
    )

    data = torch.tensor(data)
    if filtered_data is not None:
        filtered_data = torch.tensor(filtered_data)

    tsne.save_means_and_vars(data, filtered_data)
    lines = file_handle.getvalue().splitlines()

    assert lines[0].split() == ["column", "mean", "var"]
    for i, (mean, var) in enumerate(zip(data_means, data_vars)):
        assert lines[i + 1].split() == [f"{i}", f"{mean}", f"{var}"]
    if filtered_data is not None:
        assert lines[len(data_means) + 2].split() == [
            "filtered_column",
            "filtered_mean",
            "filtered_var",
        ]
        for i, (filtered_mean, filtered_var) in enumerate(
            zip(filtered_data_means, filtered_data_vars)
        ):
            assert lines[i + len(data_means) + 3].split() == [
                f"{i}",
                f"{filtered_mean}",
                f"{filtered_var}",
            ]


@pytest.mark.parametrize("data_name, step, output_path", [("test", 2, "output")])
@patch("torch.stack")
@patch("NeuralTSNE.TSNE.neural_tsne.save_torch_labels")
@patch("NeuralTSNE.TSNE.neural_tsne.save_means_and_vars")
@patch("NeuralTSNE.DatasetLoader.get_datasets.get_dataset")
def test_load_torch_dataset(
    mock_get_dataset: MagicMock,
    mock_save_means_and_vars: MagicMock,
    mock_save_torch_labels: MagicMock,
    mock_stack: MagicMock,
    data_name: str,
    step: int,
    output_path: str,
):
    train_data = torch.tensor([1, 2, 3])
    test_data = torch.tensor([1, 2, 3])
    train_data = torch.utils.data.TensorDataset(train_data)
    test_data = torch.utils.data.TensorDataset(test_data)
    mock_get_dataset.return_value = train_data, test_data

    tsne.load_torch_dataset(data_name, step, output_path)

    mock_get_dataset.assert_called_once_with(data_name)
    mock_stack.assert_called_once()
    mock_save_torch_labels.assert_called_once_with(output_path, test_data)
    mock_save_means_and_vars.assert_called_once_with(mock_stack.return_value)


@pytest.mark.parametrize("output_path", ["output.txt"])
@patch("builtins.open", new_callable=mock_open)
def test_save_torch_labels(mock_open: MagicMock, output_path: str):
    TQDM_DISABLE = 1
    file_handle = PersistentStringIO()
    mock_open.return_value = file_handle

    data = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    labels = torch.tensor([1, 2, 3])

    data_set = torch.utils.data.TensorDataset(data, labels)

    tsne.save_torch_labels(output_path, data_set)

    new_file_path = output_path.replace(".txt", "_labels.txt")
    mock_open.assert_called_once_with(new_file_path, "w")

    assert file_handle.getvalue() == "1\n2\n3\n"


@pytest.mark.parametrize(
    "labels", ["1\n2\n3\n", "3\n2\n1\n6\n", "1\n3\n2\n", "2\n1\n3\n4\n", None]
)
def test_load_labels(labels: str | None):
    if labels is None:
        assert tsne.load_labels(labels) is None
    else:
        expected = torch.tensor([float(label) for label in labels.splitlines()])
        labels_file = io.StringIO(labels)
        assert torch.allclose(tsne.load_labels(labels_file), expected)
        assert labels_file.closed


@pytest.mark.parametrize(
    "file_content, header",
    [
        ("1\t3\t7\t8\n4\t1\t3\t2\n7\t1\t9\t5\n3\t1\t12\t9\n", False),
        (
            "col1\tcol2\tcol3\tcol4\n1\t3\t7\t8\n4\t1\t3\t2\n7\t1\t9\t5\n3\t1\t12\t9\n",
            True,
        ),
        ("1 3 7 8\n4 1 3 2\n7 1 9 5\n3 1 12 9\n", False),
        ("col1 col2 col3 col4\n1 3 7 8\n4 1 3 2\n7 1 9 5\n3 1 12 9\n", True),
    ],
)
@pytest.mark.parametrize("variance_threshold", [0.1, 0.5, 0.9])
@pytest.mark.parametrize("exclude_cols", [[3], [1, 2], [3, 1], None])
@pytest.mark.parametrize("step", [1, 2, 3])
@patch("NeuralTSNE.TSNE.neural_tsne.prepare_data")
@patch("builtins.open", new_callable=mock_open)
def test_load_text_file(
    mock_open: MagicMock,
    mock_prepare_data: MagicMock,
    file_content: str,
    step: int,
    header: bool,
    exclude_cols: List[int] | None,
    variance_threshold: float,
):
    mock_content = io.StringIO(file_content)
    mock_open.return_value = mock_content
    start_index = 1 if header else 0
    matrix = np.array(
        [
            [float(value) for value in line.split()]
            for line in file_content.splitlines()[start_index:]
        ]
    )

    if exclude_cols is not None:
        cols = [col for col in range(len(matrix[0])) if col not in exclude_cols]
        matrix = matrix[:, cols]

    matrix = matrix[::step]
    matrix_t = torch.tensor(matrix).T
    mock_prepare_data.return_value = matrix_t

    result = tsne.load_text_file(
        mock_open, step, header, exclude_cols, variance_threshold
    )

    mock_prepare_data.assert_called_once()
    prepare_data_args = mock_prepare_data.call_args[0]
    assert mock_content.closed
    assert prepare_data_args[0] == variance_threshold
    assert np.allclose(prepare_data_args[1], matrix)
    assert torch.allclose(result, matrix_t)


@pytest.mark.parametrize(
    "file_content",
    ["1 2 3 4\n5 6 7 8\n9 10 11 12\n", "1\t2\t3\t4\n5\t6\t7\t8\n9\t10\t11\t12\n"],
)
@pytest.mark.parametrize("step", [1, 2, 3])
@pytest.mark.parametrize("exclude_cols", [[3], [1, 2], [3, 1], None])
@pytest.mark.parametrize("variance_threshold", [0.1, 0.5, 0.9])
@patch("NeuralTSNE.TSNE.neural_tsne.prepare_data")
def test_load_npy_file(
    mock_prepare_data: MagicMock,
    file_content: str,
    step: int,
    exclude_cols: List[int] | None,
    variance_threshold: float,
):
    matrix = np.loadtxt(io.StringIO(file_content))

    matrix_file = io.BytesIO()
    np.save(matrix_file, matrix)
    matrix_file.seek(0)

    if exclude_cols is not None:
        cols = [col for col in range(len(matrix[0])) if col not in exclude_cols]
        matrix = matrix[:, cols]

    matrix = matrix[::step]
    matrix_t = torch.tensor(matrix).T
    mock_prepare_data.return_value = matrix_t

    result = tsne.load_npy_file(matrix_file, step, exclude_cols, variance_threshold)

    mock_prepare_data.assert_called_once()
    prepare_data_args = mock_prepare_data.call_args[0]
    assert prepare_data_args[0] == variance_threshold
    assert np.allclose(prepare_data_args[1], matrix)
    assert torch.allclose(result, matrix_t)


@pytest.mark.parametrize("variance_threshold", [0.1, 0.5, None])
@pytest.mark.parametrize(
    "data",
    [[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[9, 3, 3, 1], [1, 4, 2, 6], [3, 5, 11, 9]]],
)
@patch("NeuralTSNE.TSNE.neural_tsne.normalize_columns")
@patch("NeuralTSNE.TSNE.neural_tsne.save_means_and_vars")
@patch("NeuralTSNE.TSNE.neural_tsne.filter_data_by_variance")
def test_prepare_data(
    mock_filter_data_by_variance: MagicMock,
    mock_save_means_and_vars: MagicMock,
    mock_normalize_columns: MagicMock,
    data: List[List[float]],
    variance_threshold: float | None,
):
    data = np.array(data)
    filtered = None if variance_threshold is None else data
    data_t = torch.tensor(data, dtype=torch.float32)
    mock_filter_data_by_variance.return_value = filtered
    mock_normalize_columns.return_value = data_t

    result = tsne.prepare_data(variance_threshold, data)

    mock_filter_data_by_variance.assert_called_once_with(data, variance_threshold)
    mock_normalize_columns.assert_called_once()
    normalize_columns_args = mock_normalize_columns.call_args[0]
    assert np.allclose(normalize_columns_args[0], data_t)
    mock_save_means_and_vars.assert_called_once()
    save_means_and_vars_args = mock_save_means_and_vars.call_args[0]
    np.allclose(save_means_and_vars_args[0], data)
    if variance_threshold is None:
        assert save_means_and_vars_args[1] is None
    else:
        assert np.allclose(save_means_and_vars_args[1], filtered)
    assert torch.allclose(result, data_t)


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
    H, P = tsne.Hbeta(D, beta)
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
    result = tsne.x2p_job(data, tolerance, max_iterations)
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
@patch("NeuralTSNE.TSNE.neural_tsne.x2p_job")
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

    P = tsne.x2p(X, perplexity, tolerance)
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


@pytest.fixture
def neural_network_params(
    request: type[pytest.FixtureRequest],
) -> dict[str, Any]:
    initial_features, n_components, multipliers = request.param
    return {
        "initial_features": initial_features,
        "n_components": n_components,
        "multipliers": multipliers,
    }


@pytest.fixture
def neural_network(neural_network_params: dict[str, Any]) -> tsne.NeuralNetwork:
    return tsne.NeuralNetwork(**neural_network_params)


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

    for i, layer in enumerate(neural_network.linear_relu_stack):
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

    gradient = neural_network.linear_relu_stack[0].weight
    gradient = gradient.clone().detach()

    loss.backward()

    optimizer.step()

    gradient_after = neural_network.linear_relu_stack[0].weight

    assert not torch.allclose(gradient, gradient_after, atol=1e-8)


# region Test ParametricTSNE


@pytest.fixture
def parametric_tsne_instance(request):
    params = request.param
    with (
        patch("NeuralTSNE.TSNE.neural_tsne.ParametricTSNE.set_loss_fn") as mock_loss_fn,
        patch("torchinfo.summary") as mock_summary,
        patch("NeuralTSNE.TSNE.neural_tsne.NeuralNetwork", autospec=True) as mock_nn,
    ):
        mock_loss_fn.return_value = params["loss_fn"]
        instance = tsne.ParametricTSNE(**params)
        yield instance, params, {
            "loss_fn": mock_loss_fn,
            "summary": mock_summary,
            "nn": mock_nn,
        }


@pytest.fixture
def default_parametric_tsne_instance():
    params = {
        "loss_fn": "kl_divergence",
        "n_components": 3,
        "perplexity": 50,
        "batch_size": 10,
        "early_exaggeration_epochs": 10,
        "early_exaggeration_value": 8.0,
        "max_iterations": 500,
        "features": 512,
        "multipliers": [1.0, 1.5],
        "n_jobs": 2,
        "tolerance": 1e-6,
        "force_cpu": True,
    }
    with patch("torchinfo.summary"):
        yield tsne.ParametricTSNE(**params), params


@pytest.mark.parametrize(
    "parametric_tsne_instance",
    [
        {
            "loss_fn": "mse",
            "n_components": 2,
            "perplexity": 30,
            "batch_size": 64,
            "early_exaggeration_epochs": 5,
            "early_exaggeration_value": 12.0,
            "max_iterations": 300,
            "features": 256,
            "multipliers": [1.0, 2.0],
            "n_jobs": 0,
            "tolerance": 1e-5,
            "force_cpu": False,
        },
        {
            "loss_fn": "kl_divergence",
            "n_components": 3,
            "perplexity": 50,
            "batch_size": 128,
            "early_exaggeration_epochs": 10,
            "early_exaggeration_value": 8.0,
            "max_iterations": 500,
            "features": 512,
            "multipliers": [1.0, 1.5],
            "n_jobs": 2,
            "tolerance": 1e-6,
            "force_cpu": True,
        },
    ],
    indirect=True,
)
def test_parametric_tsne_init(parametric_tsne_instance):
    tsne_instance, params, mocks = parametric_tsne_instance
    tsne_dict = tsne_instance.__dict__
    del tsne_dict["device"], tsne_dict["model"]

    mocks["loss_fn"].assert_called_once_with(params["loss_fn"])
    mocks["nn"].assert_called_once_with(
        params["features"],
        params["n_components"],
        params["multipliers"],
    )
    del (
        params["features"],
        params["n_components"],
        params["multipliers"],
        params["force_cpu"],
    )
    assert isinstance(tsne_instance, tsne.ParametricTSNE)
    assert tsne_dict == params


@pytest.mark.parametrize("loss_fn", ["kl_divergence"])
def test_set_loss_fn(default_parametric_tsne_instance, loss_fn: List[str]):
    tsne_instance, params = default_parametric_tsne_instance
    tsne_instance.loss_fn = None
    tsne_instance.set_loss_fn(loss_fn)

    assert tsne_instance.loss_fn is not None


@pytest.mark.parametrize("loss_fn", ["dummy"])
def test_set_invalid_loss_fn(default_parametric_tsne_instance, loss_fn: List[str]):
    tsne_instance, params = default_parametric_tsne_instance
    tsne_instance.loss_fn = None
    tsne_instance.set_loss_fn(loss_fn)

    assert tsne_instance.loss_fn is None


@pytest.mark.parametrize("filename", ["test", "model"])
@patch("NeuralTSNE.TSNE.neural_tsne.torch.save")
def test_save_model(
    mock_save: MagicMock, filename: str, default_parametric_tsne_instance
):
    tsne_instance, _ = default_parametric_tsne_instance
    tsne_instance.save_model(filename)

    mock_save.assert_called_once()
    args = mock_save.call_args_list[0].args
    assert isinstance(args[0], OrderedDict)
    assert args[1] == filename


@pytest.mark.parametrize("filename", ["test", "model"])
@patch("NeuralTSNE.TSNE.neural_tsne.torch.load")
@patch("NeuralTSNE.TSNE.neural_tsne.NeuralNetwork.load_state_dict")
def test_read_model(
    mock_load_dict: MagicMock,
    mock_load: MagicMock,
    filename: str,
    default_parametric_tsne_instance,
):
    tsne_instance, _ = default_parametric_tsne_instance
    tsne_instance.read_model(filename)

    mock_load.assert_called_once_with(filename)
    mock_load_dict.assert_called_once_with(mock_load(filename))


@pytest.fixture
def mock_dataloaders():
    with patch(
        "NeuralTSNE.TSNE.neural_tsne.ParametricTSNE.create_dataloaders", autospec=True
    ) as mock_create_dataloaders:
        mock_create_dataloaders.return_value = DataLoader(Dataset()), DataLoader(
            Dataset()
        )
        yield mock_create_dataloaders


@pytest.mark.parametrize(
    "split", [(0.8, 0.2), (0.6, 0.4), (0.55, 0.45), (0, 1), (1, 0)]
)
@pytest.mark.parametrize("labels", [True, False])
def test_split_dataset(
    mock_dataloaders,
    default_parametric_tsne_instance,
    split: Tuple[float, float],
    labels: bool,
):
    tsne_instance, _ = default_parametric_tsne_instance
    y = None
    X = torch.randn(100, 10)
    if labels:
        y = torch.randint(0, 2, (100,))
    train_dataloader, test_dataloader = tsne_instance.split_dataset(
        X, y, train_size=split[0], test_size=split[1]
    )

    assert isinstance(train_dataloader, DataLoader)
    assert isinstance(test_dataloader, DataLoader)
    assert mock_dataloaders.call_count == 1

    args = mock_dataloaders.call_args_list[0].args
    train = args[1]
    test = args[2]

    train_len = len(train) if split[0] != 0 else None
    test_len = len(test) if split[1] != 0 else None
    tensors_number = 1 if not labels else 2

    if train_len is None or test_len is None:
        if split[0] == 0:
            assert train is None
            assert len(test.dataset.tensors) == tensors_number
            assert test_len == X.shape[0]
        else:
            assert test is None
            assert len(train.dataset.tensors) == tensors_number
            assert train_len is X.shape[0]
    else:
        assert len(train.dataset.tensors) == tensors_number
        assert len(test.dataset.tensors) == tensors_number

        eps = 1e-4
        assert (
            split[0] - eps < train_len / (train_len + test_len) < split[0] + eps
        ) is True
        assert (
            split[1] - eps < test_len / (train_len + test_len) < split[1] + eps
        ) is True


@pytest.mark.parametrize(
    "input_values, output",
    [
        ((None, None), (0.8, 0.2)),
        ((0.7, None), (0.7, 0.3)),
        ((None, 0.4), (0.6, 0.4)),
        ((0.6, 0.4), (0.6, 0.4)),
        ((0.8, 0.5), (0.8, 0.2)),
        ((0.5, 0.8), (0.5, 0.5)),
    ],
)
def test_determine_train_test_split(
    default_parametric_tsne_instance,
    input_values: Tuple[float | None, float | None],
    output: Tuple[float, float],
):
    tsne_instance, _ = default_parametric_tsne_instance
    train_size, test_size = tsne_instance._determine_train_test_split(*input_values)
    eps = 1e-4
    assert (train_size - eps < output[0] < train_size + eps) is True
    assert (test_size - eps < output[1] < test_size + eps) is True


@pytest.mark.parametrize(
    "train_dataset",
    [TensorDataset(torch.randn(100, 10), torch.randint(0, 2, (100,))), None],
)
@pytest.mark.parametrize(
    "test_dataset",
    [TensorDataset(torch.randn(20, 10), torch.randint(0, 2, (20,))), None],
)
def test_create_dataloaders(
    default_parametric_tsne_instance,
    train_dataset: TensorDataset | None,
    test_dataset: TensorDataset | None,
):
    tsne_instance, _ = default_parametric_tsne_instance
    train_loader, test_loader = tsne_instance.create_dataloaders(
        train_dataset, test_dataset
    )

    if train_dataset is None:
        assert train_loader is None
    else:
        assert isinstance(train_loader, DataLoader)

    if test_dataset is None:
        assert test_loader is None
    else:
        assert isinstance(test_loader, DataLoader)


def test_calculate_P(default_parametric_tsne_instance):
    TQDM_DISABLE = 1
    tsne_instance, params = default_parametric_tsne_instance

    dataloader = DataLoader(
        TensorDataset(torch.randn(50, 15), torch.randint(0, 2, (50,))),
        batch_size=params["batch_size"],
    )

    result_P = tsne_instance._calculate_P(dataloader)

    assert result_P.shape == (50, params["batch_size"])


@pytest.mark.parametrize("fill_with", [0, "NaN"])
@patch("NeuralTSNE.TSNE.neural_tsne.x2p")
def test_calculate_P_mocked(
    mock_x2p: MagicMock, default_parametric_tsne_instance, fill_with: Any
):
    TQDM_DISABLE = 1
    tsne_instance, params = default_parametric_tsne_instance
    samples = 50

    dataloader = DataLoader(
        TensorDataset(torch.randn(samples, 15), torch.randint(0, 2, (50,))),
        batch_size=params["batch_size"],
    )

    select_one = random.randint(0, params["batch_size"] - 1)
    if fill_with == 0:
        ret = torch.zeros(params["batch_size"], params["batch_size"])
    if fill_with == "NaN":
        ret = torch.full((params["batch_size"], params["batch_size"]), torch.nan)

    ret[select_one] = 1
    mock_x2p.return_value = ret

    result_P = tsne_instance._calculate_P(dataloader)

    assert result_P.shape == (samples, params["batch_size"])

    expected_tensor = torch.zeros(samples, params["batch_size"])
    expected_tensor[:, select_one] = 1
    for i in range(0, samples, params["batch_size"]):
        cut = expected_tensor[i : i + params["batch_size"]]
        cut = cut + cut.T
        cut = cut / cut.sum()
        expected_tensor[i : i + params["batch_size"]] = cut
    assert torch.allclose(result_P, expected_tensor)


@pytest.mark.parametrize("fill_with", [0, "NaN"])
@patch("NeuralTSNE.TSNE.neural_tsne.x2p")
def test_calculate_P_mocked_nan(
    mock_x2p: MagicMock, default_parametric_tsne_instance, fill_with: Any
):
    TQDM_DISABLE = 1
    tsne_instance, params = default_parametric_tsne_instance
    samples = 50

    dataloader = DataLoader(
        TensorDataset(torch.randn(samples, 15), torch.randint(0, 2, (50,))),
        batch_size=params["batch_size"],
    )

    if fill_with == 0:
        ret = torch.zeros(params["batch_size"], params["batch_size"])
    if fill_with == "NaN":
        ret = torch.full((params["batch_size"], params["batch_size"]), torch.nan)

    mock_x2p.return_value = ret

    result_P = tsne_instance._calculate_P(dataloader)

    assert result_P.shape == (samples, params["batch_size"])
    assert torch.allclose(
        result_P, torch.full((samples, params["batch_size"]), torch.nan), equal_nan=True
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
    C = tsne_instance._kl_divergence(Q, P)

    assert torch.allclose(C, expected, rtol=1e-3)


# endregion

# TODO: Test Classifier


@pytest.fixture
def classifier_instance(request, default_parametric_tsne_instance):
    params = request.param

    with patch(
        "NeuralTSNE.TSNE.neural_tsne.Classifier.reset_exaggeration_status"
    ) as mock_exaggeration_status:
        yield tsne.Classifier(
            tsne=default_parametric_tsne_instance[0], **params
        ), params | {
            "tsne": default_parametric_tsne_instance[0]
        }, mock_exaggeration_status


@pytest.mark.parametrize(
    "classifier_instance",
    [{"shuffle": False, "optimizer": "adam", "lr": 1e-5}],
    indirect=True,
)
def test_classifier_init(classifier_instance):
    classifier_instance, params, mock_exaggeration_status = classifier_instance

    assert isinstance(classifier_instance, tsne.Classifier)
    assert classifier_instance.tsne == params["tsne"]
    assert classifier_instance.batch_size == params["tsne"].batch_size
    assert classifier_instance.model == params["tsne"].model
    assert classifier_instance.loss_fn == params["tsne"].loss_fn
    assert (
        classifier_instance.exaggeration_epochs
        == params["tsne"].early_exaggeration_epochs
    )
    assert (
        classifier_instance.exaggeration_value
        == params["tsne"].early_exaggeration_value
    )
    assert classifier_instance.shuffle == params["shuffle"]
    assert classifier_instance.lr == params["lr"]
    assert classifier_instance.optimizer == params["optimizer"]
    assert mock_exaggeration_status.call_count == 1


# region Test FileTypeWithExtensionCheck


@pytest.fixture
def valid_temp_file(request):
    file = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
    temp_file_path = file.name
    file.close()
    yield temp_file_path
    os.remove(temp_file_path)


# TODO: Parametrize fixture to test multiple valid extensions through request


@pytest.fixture
def invalid_temp_file(request):
    file = tempfile.NamedTemporaryFile(suffix=".bat", delete=False)
    temp_file_path = file.name
    file.close()
    yield temp_file_path
    os.remove(temp_file_path)


def test_valid_extension(valid_temp_file: str):
    file_type = tsne.FileTypeWithExtensionCheck(valid_extensions="txt")
    result = file_type(valid_temp_file)
    assert result.name == valid_temp_file


def test_invalid_extension(invalid_temp_file: str):
    file_type = tsne.FileTypeWithExtensionCheck(valid_extensions="txt")
    with pytest.raises(argparse.ArgumentTypeError):
        file_type(invalid_temp_file)


@pytest.mark.parametrize(
    "temp_file_fixture",
    [
        pytest.param("valid_temp_file", marks=pytest.mark.valid_file),
        pytest.param("invalid_temp_file", marks=pytest.mark.invalid_file),
    ],
)
def test_no_extension_check(temp_file_fixture: str, request):
    file_type = tsne.FileTypeWithExtensionCheck()
    temp_file_path = request.getfixturevalue(temp_file_fixture)
    result = file_type(temp_file_path)
    assert result.name == temp_file_path


# endregion


# region Test FileTypeWithExtensionCheckWithPredefinedDatasets


@pytest.fixture
def file_type_with_datasets(request):
    available_datasets = request.param.get("available_datasets", [])
    return tsne.FileTypeWithExtensionCheckWithPredefinedDatasets(
        valid_extensions="txt", available_datasets=available_datasets
    )


def test_valid_extension_with_datasets(valid_temp_file: str):
    file_type = tsne.FileTypeWithExtensionCheckWithPredefinedDatasets(
        valid_extensions="txt"
    )
    result = file_type(valid_temp_file)
    assert result.name == valid_temp_file


def test_invalid_extension_with_datasets(invalid_temp_file: str):
    file_type = tsne.FileTypeWithExtensionCheckWithPredefinedDatasets(
        valid_extensions="txt"
    )
    with pytest.raises(argparse.ArgumentTypeError):
        file_type(invalid_temp_file)


@pytest.mark.parametrize(
    "temp_file_fixture",
    [
        pytest.param("valid_temp_file", marks=pytest.mark.valid_file),
        pytest.param("invalid_temp_file", marks=pytest.mark.invalid_file),
    ],
)
def test_no_extension_check_with_datasets(temp_file_fixture: str, request):
    file_type = tsne.FileTypeWithExtensionCheckWithPredefinedDatasets()
    temp_file_path = request.getfixturevalue(temp_file_fixture)
    result = file_type(temp_file_path)
    assert result.name == temp_file_path


@pytest.mark.parametrize("dataset", ["dataset1", "dataset2"])
@pytest.mark.parametrize("available_datasets", [["dataset1", "dataset2"]])
def test_predefined_dataset(available_datasets: List[str], dataset: str):
    file_type = tsne.FileTypeWithExtensionCheckWithPredefinedDatasets(
        available_datasets=available_datasets
    )
    result = file_type(dataset)
    assert result == dataset


@pytest.mark.parametrize("dataset", ["dataset3", "invalid_dataset"])
@pytest.mark.parametrize("available_datasets", [["dataset1", "dataset2"]])
def test_invalid_dataset(available_datasets: List[str], dataset: str):
    file_type = tsne.FileTypeWithExtensionCheckWithPredefinedDatasets(
        valid_extensions="txt", available_datasets=available_datasets
    )
    with pytest.raises(argparse.ArgumentTypeError):
        file_type(dataset)


@pytest.mark.parametrize("dataset", ["dataset3", "invalid_dataset"])
@pytest.mark.parametrize("available_datasets", [["dataset1", "dataset2"]])
def test_invalid_dataset_with_no_extension(available_datasets: List[str], dataset: str):
    file_type = tsne.FileTypeWithExtensionCheckWithPredefinedDatasets(
        available_datasets=available_datasets
    )
    with pytest.raises(argparse.ArgumentTypeError):
        file_type(dataset)


# endregion


@pytest.fixture(params=[""])
def get_output_filename(request):
    suffix = request.param
    file_name = get_random_string(10)
    yield file_name, suffix
    file_to_delete = f"{file_name}{suffix}"
    if os.path.exists(file_to_delete):
        os.remove(file_to_delete)


@pytest.mark.parametrize("batch_shape", [(5, 2), None])
@pytest.mark.parametrize("step", [30, 45, 2])
def test_save_results(
    get_output_filename: str, batch_shape: Tuple[int | int] | None, step: int
):
    TQDM_DISABLE = 1
    filename, _ = get_output_filename
    args = {"o": filename, "step": step}

    test_data = None

    if batch_shape:
        num_samples = batch_shape[0] * 10
        dataset = MyDataset(num_samples, batch_shape[1])
        test_data = DataLoaderMock(dataset, batch_size=2)

    entries_num = random.randint(20, 500)
    Y = [
        [(random.random(), random.random()) for _ in range(entries_num)]
        for _ in range(2)
    ]

    tsne.save_results(args, test_data, Y)

    if batch_shape is None:
        assert os.path.exists(filename) is False
        return

    with open(filename, "r") as f:
        lines = f.readlines()

    assert lines[0].strip() == str(step)
    expected_lines = ["\t".join(tuple(map(str, item))) for batch in Y for item in batch]
    for i in range(1, len(lines)):
        assert lines[i].strip() == expected_lines[i - 1]


@pytest.mark.parametrize(
    "get_output_filename",
    ["_labels.txt"],
    indirect=["get_output_filename"],
)
@pytest.mark.parametrize("num_batches", [3, 5])
@pytest.mark.parametrize("batch_shape", [(5, 3), (4, 4), None])
def test_save_labels_data(
    get_output_filename: str, num_batches: int, batch_shape: Tuple[int, int] | None
):
    TQDM_DISABLE = 1
    filename, suffix = get_output_filename
    args = {"o": filename}

    test_data = None

    if batch_shape:
        num_samples = batch_shape[0] * 10
        dataset = MyDataset(num_samples, batch_shape[1], (0, 10), True)
        test_data = DataLoaderMock(dataset, batch_size=num_batches)

    tsne.save_labels_data(args, test_data)

    if batch_shape:
        data = [
            "\t".join(map(str, row.tolist()))
            for batch in test_data.batches
            for tensor in batch
            for row in tensor
        ]

    if batch_shape is None:
        assert os.path.exists(filename) is False
        return

    with open(f"{filename}{suffix}", "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        assert line.strip() == data[i]
