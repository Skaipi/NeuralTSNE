import io
from typing import List
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest
import torch

from NeuralTSNE.Utils.Loaders.FileLoaders import (
    load_npy_file,
    load_text_file,
    load_torch_dataset,
)


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
@patch("NeuralTSNE.Utils.Loaders.FileLoaders.file_loaders.prepare_data")
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

    result = load_text_file(mock_open, step, header, exclude_cols, variance_threshold)

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
@patch("NeuralTSNE.Utils.Loaders.FileLoaders.file_loaders.prepare_data")
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

    result = load_npy_file(matrix_file, step, exclude_cols, variance_threshold)

    mock_prepare_data.assert_called_once()
    prepare_data_args = mock_prepare_data.call_args[0]
    assert prepare_data_args[0] == variance_threshold
    assert np.allclose(prepare_data_args[1], matrix)
    assert torch.allclose(result, matrix_t)


@pytest.mark.parametrize("data_name, step, output_path", [("test", 2, "output")])
@patch("torch.stack")
@patch("NeuralTSNE.Utils.Loaders.FileLoaders.file_loaders.save_torch_labels")
@patch("NeuralTSNE.Utils.Loaders.FileLoaders.file_loaders.save_means_and_vars")
@patch("NeuralTSNE.Utils.Loaders.FileLoaders.file_loaders.get_datasets.get_dataset")
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

    load_torch_dataset(data_name, step, output_path)

    mock_get_dataset.assert_called_once_with(data_name)
    mock_stack.assert_called_once()
    mock_save_torch_labels.assert_called_once_with(output_path, test_data)
    mock_save_means_and_vars.assert_called_once_with(mock_stack.return_value)
