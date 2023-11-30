import itertools
from unittest.mock import MagicMock, call, patch

import pytest
from parameterized import parameterized

from NeuralTSNE.DatasetLoader import get_datasets as loader


@patch("torchvision.datasets.MNIST")
def test_get_mnist(mock_mnist: MagicMock):
    mock_mnist.side_effect = ["mocked mnist train", "mocked mnist test"]
    assert loader.get_mnist() == ("mocked mnist train", "mocked mnist test")
    assert mock_mnist.call_count == 2


@patch("torchvision.datasets.FashionMNIST")
def test_get_fashion_mnist(mock_fashion_mnist: MagicMock):
    mock_fashion_mnist.side_effect = [
        "mocked fashion mnist train",
        "mocked fashion mnist test",
    ]
    assert loader.get_fashion_mnist() == (
        "mocked fashion mnist train",
        "mocked fashion mnist test",
    )
    assert mock_fashion_mnist.call_count == 2


def test_get_available_datasets():
    l_dict = loader.__dict__
    available_datasets = [key[4:] for key in l_dict if key.startswith("get")]
    available_datasets.remove("dataset")
    assert loader._get_available_datasets() == available_datasets


@parameterized.expand(
    itertools.product(
        ["mnist", "fashion_mnist", "abcdef"], [True, False], [True, False]
    )
)
@patch("torch.load")
@patch("torch.save")
@patch("NeuralTSNE.DatasetLoader.get_datasets.os.path.exists")
def test_prepare_dataset(
    dataset: str,
    train_exists: bool,
    test_exists: bool,
    mock_exists: MagicMock,
    mock_save: MagicMock,
    mock_load: MagicMock,
):
    mock_exists.side_effect = [train_exists, test_exists]
    if train_exists and test_exists:
        mock_load.side_effect = ["mocked train", "mocked test"]
        assert loader.prepare_dataset(dataset) == ("mocked train", "mocked test")
        assert mock_load.call_count == 2
        assert mock_save.call_count == 0
        mock_load.assert_has_calls(
            [call(dataset + "_train.data"), call(dataset + "_test.data")]
        )
    else:
        l_dict = loader.__dict__
        available_datasets = [key[4:] for key in l_dict if key.startswith("get")]
        available_datasets.remove("dataset")
        if dataset not in available_datasets:
            with pytest.raises(KeyError):
                loader.prepare_dataset(dataset)
        else:
            with patch(
                "NeuralTSNE.DatasetLoader.get_datasets.get_" + dataset
            ) as mock_get:
                mock_get.return_value = ("mocked train", "mocked test")
                assert loader.prepare_dataset(dataset) == (
                    "mocked train",
                    "mocked test",
                )
                assert mock_get.call_count == 1
                assert mock_save.call_count == 2
                mock_save.assert_has_calls(
                    [
                        call("mocked train", dataset + "_train.data"),
                        call("mocked test", dataset + "_test.data"),
                    ]
                )


@parameterized.expand(itertools.product(["mnist"], [True, False]))
@patch("NeuralTSNE.DatasetLoader.get_datasets.prepare_dataset")
@patch("NeuralTSNE.DatasetLoader.get_datasets._get_available_datasets")
def test_get_dataset(
    dataset: str, is_available: bool, mock_available: MagicMock, mock_prepare: MagicMock
):
    if not is_available:
        mock_available.return_value = []
    else:
        mock_available.return_value = [dataset]
        mock_prepare.return_value = ("mocked train", "mocked test")

    returned = loader.get_dataset(dataset)

    if not is_available:
        mock_prepare.assert_not_called()
        assert returned == (None, None)
    else:
        mock_prepare.assert_called_once_with(dataset)
        assert returned == ("mocked train", "mocked test")
