import numpy as np
import pytest
import torch

from NeuralTSNE.Utils.Preprocessing.Filters.filters import filter_data_by_variance


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

    result = filter_data_by_variance(artificial_data, variance_threshold)
    assert torch.allclose(result, filtered_data)
