import torch


def normalize_columns(data: torch.Tensor) -> torch.Tensor:
    """Normalize the columns of a 2D `torch.Tensor` to have values in the range `[0, 1]`.

    Parameters
    ----------
    `data` : `torch.Tensor`
        The input 2D tensor with columns to be normalized.

    Returns
    -------
    `torch.Tensor`
        A new tensor with columns normalized to the range `[0, 1]`.

    Note
    ----
    The normalization is done independently for each column, ensuring that the values in each column are scaled to the range `[0, 1]`.
    """
    data_min = data.min(dim=0)[0]
    data_range = data.max(dim=0)[0] - data_min
    return (data - data_min) / data_range
