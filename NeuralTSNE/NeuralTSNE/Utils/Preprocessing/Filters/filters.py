from typing import Union

import numpy as np
import torch


def filter_data_by_variance(
    data: torch.Tensor, variance_threshold: float
) -> Union[torch.Tensor, None]:
    """
    Filter columns of a 2D `torch.Tensor` based on the variance of each column.

    If the `variance_threshold` is `None`, the function returns `None`, indicating no filtering is performed.

    ---
    ### Parameters:
        - `data` (`torch.Tensor`): The input 2D tensor with columns to be filtered.
        - `variance_threshold` (`float`): The threshold for column variance. Columns with variance below this threshold will be filtered out.

    ---
    ### Returns:
        - `Union[torch.Tensor, None]`: If `variance_threshold` is `None`, returns `None`. Otherwise, returns a new `tensor` with columns filtered based on variance.

    ---
    ### Note:
    - If `variance_threshold` is set to `None`, the function returns `None`, and no filtering is performed.
    - The function filters columns based on the variance of each column, keeping only those with variance greater than the specified threshold.
    """
    if variance_threshold is None:
        return None
    column_vars = data.var(axis=0)
    cols = np.where(column_vars > variance_threshold)[0]
    filtered_data = data[:, cols]
    return filtered_data
