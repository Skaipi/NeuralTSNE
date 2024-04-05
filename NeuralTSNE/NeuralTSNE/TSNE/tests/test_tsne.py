import io
import random
from collections import OrderedDict
from typing import Any, List, Tuple
from unittest.mock import MagicMock, patch

import lightning as L
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

import NeuralTSNE.TSNE as tsne
from NeuralTSNE.TSNE.cost_functions import CostFunctions


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
        patch(
            "NeuralTSNE.TSNE.parametric_tsne.ParametricTSNE.set_loss_fn"
        ) as mock_loss_fn,
        patch("torchinfo.summary") as mock_summary,
        patch(
            "NeuralTSNE.TSNE.parametric_tsne.NeuralNetwork", autospec=True
        ) as mock_nn,
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
    with pytest.raises(AttributeError):
        tsne_instance.set_loss_fn(loss_fn)


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
@patch("NeuralTSNE.TSNE.neural_network.NeuralNetwork.load_state_dict")
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
@patch("NeuralTSNE.TSNE.parametric_tsne.x2p")
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
@patch("NeuralTSNE.TSNE.parametric_tsne.x2p")
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
)  # TODO: EXTRACT LOSS_FN TO A SEPARATE FILE
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


# endregion

# region Test Classifier


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


@pytest.fixture(params=[None])
def default_classifier_instance(request, default_parametric_tsne_instance):
    tsne_params = request.param or {}
    params = {"shuffle": True, "optimizer": "rmsprop", "lr": 1e-6}
    tsne_instance, default_tsne_params = default_parametric_tsne_instance
    for k, v in tsne_params.items():
        tsne_instance.__dict__[k] = v
    return tsne.Classifier(tsne=tsne_instance, **params), {
        "tsne_params": tsne_params,
        "default_tsne_params": default_tsne_params,
        "params": params,
    }


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


@pytest.mark.parametrize(
    "default_classifier_instance",
    [{"early_exaggeration_epochs": 0}, {"early_exaggeration_epochs": 10}],
    indirect=True,
)
def test_reset_exaggeration_status(default_classifier_instance):
    classifier_instance, params = default_classifier_instance
    classifier_instance.reset_exaggeration_status()

    params = params["tsne_params"]
    if params["early_exaggeration_epochs"] == 0:
        assert classifier_instance.has_exaggeration_ended == True
    else:
        assert classifier_instance.has_exaggeration_ended == False


@pytest.mark.parametrize(
    "optimizer, expected_instance",
    [
        ("adam", torch.optim.Adam),
        ("sgd", torch.optim.SGD),
        ("rmsprop", torch.optim.RMSprop),
    ],
)
def test_set_optimizer(
    default_classifier_instance,
    optimizer: str,
    expected_instance: torch.optim.Optimizer,
):
    classifier_instance, _ = default_classifier_instance

    returned = classifier_instance._set_optimizer(
        optimizer, {"lr": classifier_instance.lr}
    )
    assert isinstance(returned, expected_instance)
    assert returned.param_groups[0]["lr"] == classifier_instance.lr


@pytest.mark.parametrize("optimizer", ["dummy_optimizer", "adom"])
def test_set_optimizer_invalid(default_classifier_instance, optimizer: str):
    classifier_instance, _ = default_classifier_instance

    with pytest.raises(ValueError):
        classifier_instance._set_optimizer(optimizer, {"lr": classifier_instance.lr})


def test_predict_step(default_classifier_instance):
    classifier_instance, params = default_classifier_instance
    tsne_instance = classifier_instance.tsne
    num_samples = tsne_instance.batch_size * 10
    dataset = MyDataset(num_samples, 15)
    test_data = DataLoaderMock(dataset, batch_size=tsne_instance.batch_size)

    for i, batch in enumerate(test_data):
        logits = classifier_instance.predict_step(batch, i)
        assert logits.shape == (
            tsne_instance.batch_size,
            params["default_tsne_params"]["n_components"],
        )


@pytest.mark.parametrize("has_P_multiplied", [True, False])
@pytest.mark.parametrize("has_exaggeration_ended", [True, False])
def test_on_train_epoch_end(
    default_classifier_instance, has_P_multiplied: bool, has_exaggeration_ended: bool
):
    classifier_instance, _ = default_classifier_instance

    if has_P_multiplied:
        classifier_instance.P_multiplied = torch.tensor(torch.nan)
    classifier_instance.has_exaggeration_ended = has_exaggeration_ended

    classifier_instance.on_train_epoch_end()

    if has_P_multiplied:
        assert (
            hasattr(classifier_instance, "P_multiplied") is not has_exaggeration_ended
        )
    else:
        assert hasattr(classifier_instance, "P_multiplied") is False


@pytest.mark.parametrize("has_P", [True, False])
def test_on_train_start(default_classifier_instance, has_P: bool):
    classifier_instance, _ = default_classifier_instance
    tsne_instance = classifier_instance.tsne
    num_samples = tsne_instance.batch_size * 10
    dataset = MyDataset(num_samples, 15)
    test_data = DataLoaderMock(dataset, batch_size=tsne_instance.batch_size)

    trainer = L.Trainer(fast_dev_run=True)

    if has_P:
        classifier_instance.P = torch.tensor(torch.nan)

    with (
        patch.object(tsne.ParametricTSNE, "_calculate_P") as mocked_calculate_P,
        patch.object(
            tsne.Classifier, "training_step", autospec=True
        ) as mocked_training_step,
        patch.object(tsne.Classifier, "on_train_epoch_start"),
        patch.object(tsne.Classifier, "on_train_epoch_end"),
    ):
        mocked_calculate_P.return_value = torch.tensor(torch.nan)
        mocked_training_step.return_value = None

        trainer.fit(classifier_instance, test_data)

    if not has_P:
        assert mocked_calculate_P.call_count == 1
    else:
        assert mocked_calculate_P.call_count == 0

    assert torch.allclose(
        classifier_instance.P, torch.tensor(torch.nan), equal_nan=True
    )


@pytest.mark.parametrize("epochs", [1, 2, 3])
@pytest.mark.parametrize("has_exaggeration_ended", [True, False])
@pytest.mark.parametrize("exaggeration_epochs", [0, 1])
def test_on_train_epoch_start(
    default_classifier_instance,
    epochs: int,
    has_exaggeration_ended: bool,
    exaggeration_epochs: int,
):
    classifier_instance, params = default_classifier_instance

    tsne_instance = classifier_instance.tsne
    num_samples = tsne_instance.batch_size * 10
    dataset = MyDataset(num_samples, 15)
    test_data = DataLoaderMock(dataset, batch_size=tsne_instance.batch_size)

    trainer = L.Trainer(max_epochs=epochs, limit_train_batches=1)

    input_P = torch.ones((num_samples, tsne_instance.batch_size))
    classifier_instance.P = input_P

    classifier_instance.has_exaggeration_ended = has_exaggeration_ended
    classifier_instance.exaggeration_epochs = exaggeration_epochs

    with (
        patch.object(tsne.Classifier, "on_train_start"),
        patch.object(
            tsne.Classifier, "training_step", autospec=True
        ) as mocked_training_step,
        patch.object(tsne.Classifier, "on_train_epoch_end"),
    ):
        mocked_training_step.return_value = None

        trainer.fit(classifier_instance, test_data)

    if has_exaggeration_ended and exaggeration_epochs == 0:
        assert torch.allclose(classifier_instance.P_current, input_P)
    elif has_exaggeration_ended:
        assert torch.allclose(
            classifier_instance.P_current,
            input_P * params["default_tsne_params"]["early_exaggeration_value"],
        )

    if (
        not has_exaggeration_ended
        and epochs <= exaggeration_epochs
        and exaggeration_epochs > 0
    ):
        assert torch.allclose(
            classifier_instance.P_current,
            input_P * params["default_tsne_params"]["early_exaggeration_value"],
        )
    elif not has_exaggeration_ended:
        assert torch.allclose(classifier_instance.P_current, input_P)
        assert classifier_instance.has_exaggeration_ended is True


def test_training_step(default_classifier_instance):
    classifier_instance, params = default_classifier_instance

    tsne_instance = classifier_instance.tsne
    num_samples = tsne_instance.batch_size * 10
    dataset = MyDataset(num_samples, 15)
    test_data = DataLoaderMock(dataset, batch_size=tsne_instance.batch_size)

    trainer = L.Trainer(fast_dev_run=True, accelerator="cpu")

    input_P = torch.ones((num_samples, tsne_instance.batch_size))
    classifier_instance.P = input_P

    with patch.object(tsne.Classifier, "on_train_start"):
        trainer.fit(classifier_instance, test_data)


@pytest.mark.parametrize(
    "optimizer, expected_instance",
    [
        ("adam", torch.optim.Adam),
        ("sgd", torch.optim.SGD),
        ("rmsprop", torch.optim.RMSprop),
    ],
)
def test_configure_optimizers(
    default_classifier_instance,
    optimizer: str,
    expected_instance: torch.optim.Optimizer,
):
    classifier_instance, _ = default_classifier_instance
    classifier_instance.optimizer = optimizer

    returned = classifier_instance.configure_optimizers()
    assert isinstance(returned, expected_instance)
    assert returned.param_groups[0]["lr"] == classifier_instance.lr


@pytest.mark.parametrize("optimizer", ["dummy_optimizer", "adom"])
def test_configure_optimizers_invalid(default_classifier_instance, optimizer: str):
    classifier_instance, _ = default_classifier_instance
    classifier_instance.optimizer = optimizer

    with pytest.raises(ValueError):
        classifier_instance.configure_optimizers()


# endregion
