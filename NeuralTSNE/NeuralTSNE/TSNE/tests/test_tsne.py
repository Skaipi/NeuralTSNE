from unittest.mock import MagicMock, call, patch, mock_open

import pytest
import NeuralTSNE.TSNE as tsne
import numpy as np
import io

import torch


class PersistentStringIO(io.StringIO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._closed = False

    def close(self):
        self._closed = True

    @property
    def closed(self):
        return self._closed


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


# TODO: Add test for save_means_and_vars


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
