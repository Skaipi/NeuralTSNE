import random
from collections import OrderedDict
from typing import Any, List, Tuple
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from NeuralTSNE.TSNE.ParametricTSNE import ParametricTSNE

from NeuralTSNE.TSNE.tests.fixtures.dataloader_fixtures import mock_dataloaders
from NeuralTSNE.TSNE.tests.fixtures.parametric_tsne_fixtures import (
    parametric_tsne_instance,
    default_parametric_tsne_instance,
)


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
    assert isinstance(tsne_instance, ParametricTSNE)
    assert tsne_dict == params


@pytest.mark.parametrize("loss_fn", ["kl_divergence"])
def test_set_loss_fn(default_parametric_tsne_instance, loss_fn: List[str]):
    tsne_instance, _ = default_parametric_tsne_instance
    tsne_instance.loss_fn = None
    tsne_instance.set_loss_fn(loss_fn)

    assert tsne_instance.loss_fn is not None


@pytest.mark.parametrize("loss_fn", ["dummy"])
def test_set_invalid_loss_fn(default_parametric_tsne_instance, loss_fn: List[str]):
    tsne_instance, _ = default_parametric_tsne_instance
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
@patch("NeuralTSNE.TSNE.ParametricTSNE.parametric_tsne.torch.load")
@patch("NeuralTSNE.TSNE.NeuralNetwork.neural_network.NeuralNetwork.load_state_dict")
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
@patch("NeuralTSNE.TSNE.ParametricTSNE.parametric_tsne.x2p")
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
@patch("NeuralTSNE.TSNE.ParametricTSNE.parametric_tsne.x2p")
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
