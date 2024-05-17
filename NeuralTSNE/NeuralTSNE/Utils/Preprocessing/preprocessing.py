import numpy as np
import torch

from NeuralTSNE.Utils.Preprocessing.Normalizers import normalize_columns
from NeuralTSNE.Utils.Preprocessing.Filters import (
    filter_data_by_variance,
)
from NeuralTSNE.Utils.Writers.StatWriters import save_means_and_vars


def prepare_data(variance_threshold: float, data: np.ndarray) -> torch.Tensor:
    """
    Prepare data for further analysis by filtering based on variance,
    saving means and variances, and normalizing columns.

    Parameters
    ----------
    `variance_threshold` : `float`
        Threshold for variance-based feature selection.
    `data` : `np.ndarray`
        Input data array.

    Returns
    -------
    `torch.Tensor`
        Processed and normalized data tensor.

    Note
    ----
    The function filters the input `data` based on the provided `variance threshold`,
    saves means and variances, and then normalizes the columns of the `data` before
    converting it into a `torch.Tensor`.
    """
    filtered = filter_data_by_variance(data, variance_threshold)
    save_means_and_vars(data, filtered)
    if filtered is not None:
        data = filtered

    data = torch.from_numpy(data).float()
    data = normalize_columns(data)
    return data
