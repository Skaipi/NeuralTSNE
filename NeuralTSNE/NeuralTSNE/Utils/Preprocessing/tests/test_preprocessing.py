from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from NeuralTSNE.Utils.Preprocessing import prepare_data


@pytest.mark.parametrize("variance_threshold", [0.1, 0.5, None])
@pytest.mark.parametrize(
    "data",
    [[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[9, 3, 3, 1], [1, 4, 2, 6], [3, 5, 11, 9]]],
)
@patch("NeuralTSNE.Utils.Preprocessing.preprocessing.normalize_columns")
@patch("NeuralTSNE.Utils.Preprocessing.preprocessing.save_means_and_vars")
@patch("NeuralTSNE.Utils.Preprocessing.preprocessing.filter_data_by_variance")
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

    result = prepare_data(variance_threshold, data)

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
