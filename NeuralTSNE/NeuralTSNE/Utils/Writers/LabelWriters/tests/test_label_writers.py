import os
import random
import string
from typing import Tuple
from unittest.mock import MagicMock, mock_open, patch

import pytest
import torch

from NeuralTSNE.TSNE.tests.test_tsne import (
    DataLoaderMock,
    MyDataset,
    PersistentStringIO,
)
from NeuralTSNE.Utils.Writers.LabelWriters.label_writers import (
    save_labels_data,
    save_torch_labels,
)


def get_random_string(length: int):
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


@pytest.fixture(params=[""])
def get_output_filename(request):
    suffix = request.param
    file_name = get_random_string(10)
    yield file_name, suffix
    file_to_delete = f"{file_name}{suffix}"
    if os.path.exists(file_to_delete):
        os.remove(file_to_delete)


@pytest.mark.parametrize("output_path", ["output.txt"])
@patch("builtins.open", new_callable=mock_open)
def test_save_torch_labels(mock_open: MagicMock, output_path: str):
    TQDM_DISABLE = 1
    file_handle = PersistentStringIO()
    mock_open.return_value = file_handle

    data = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    labels = torch.tensor([1, 2, 3])

    data_set = torch.utils.data.TensorDataset(data, labels)

    save_torch_labels(output_path, data_set)

    new_file_path = output_path.replace(".txt", "_labels.txt")
    mock_open.assert_called_once_with(new_file_path, "w")

    assert file_handle.getvalue() == "1\n2\n3\n"


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

    save_labels_data(args, test_data)

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
