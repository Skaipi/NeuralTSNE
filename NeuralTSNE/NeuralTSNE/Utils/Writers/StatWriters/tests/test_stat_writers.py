import os
import random
import string
from typing import List, Tuple
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest
import torch

from NeuralTSNE.TSNE.tests.common import (
    DataLoaderMock,
    MyDataset,
    PersistentStringIO,
)
from NeuralTSNE.Utils.Writers.StatWriters.stat_writers import (
    save_means_and_vars,
    save_results,
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

    save_means_and_vars(data, filtered_data)
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

    save_results(args, test_data, Y)

    if batch_shape is None:
        assert os.path.exists(filename) is False
        return

    with open(filename, "r") as f:
        lines = f.readlines()

    assert lines[0].strip() == str(step)
    expected_lines = ["\t".join(tuple(map(str, item))) for batch in Y for item in batch]
    for i in range(1, len(lines)):
        assert lines[i].strip() == expected_lines[i - 1]
