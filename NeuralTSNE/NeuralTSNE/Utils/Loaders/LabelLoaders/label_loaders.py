import io
from typing import Union

import numpy as np
import torch


def load_labels(labels: io.TextIOWrapper) -> Union[torch.Tensor, None]:
    """
    Load labels from a text file into a `torch.Tensor`.

    The function reads labels from the provided text file and converts them into a `torch.Tensor` of type `float`.
    If the `labels` parameter is not provided or the file is empty, the function returns `None`.

    Parameters
    ----------
    `labels` : `io.TextIOWrapper`
        The `file` object containing labels to be loaded.

    Returns
    -------
    `torch.Tensor` | `None`
        A `torch.Tensor` containing loaded labels or `None` if no labels are available.

    Note
    ----
    - The function expects the `labels` parameter to be a file object (`io.TextIOWrapper`) with labels in text format.
    - If the file is not provided or is empty, the function returns `None`.
    - The labels are read from the file using `numpy` and then converted to a `torch.Tensor` of type `float`.
    """
    read_labels = None
    if labels:
        read_labels = np.loadtxt(labels)
        read_labels = torch.from_numpy(read_labels).float()
        labels.close()
    return read_labels
