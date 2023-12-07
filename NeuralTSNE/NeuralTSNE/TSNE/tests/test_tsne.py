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
