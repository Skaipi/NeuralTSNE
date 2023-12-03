from unittest.mock import MagicMock, call, patch, mock_open

import pytest
import NeuralTSNE.TSNE as tsne
import numpy as np
import io

import torch
from typing import List


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


@pytest.mark.parametrize(
    "labels", ["1\n2\n3\n", "3\n2\n1\n6\n", "1\n3\n2\n", "2\n1\n3\n4\n", None]
)
def test_load_labels(labels: str | None):
    if labels is None:
        assert tsne.load_labels(labels) == None
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
def test_load_text_file(
    mock_prepare_data: MagicMock,
    file_content: str,
    step: int,
    header: bool,
    exclude_cols: List[int] | None,
    variance_threshold: float,
):
    mock_file = io.StringIO(file_content)
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
        mock_file, step, header, exclude_cols, variance_threshold
    )

    mock_prepare_data.assert_called_once()
    prepare_data_args = mock_prepare_data.call_args[0]
    assert mock_file.closed
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
    assert matrix_file.closed
    assert prepare_data_args[0] == variance_threshold
    assert np.allclose(prepare_data_args[1], matrix)
    assert torch.allclose(result, matrix_t)
