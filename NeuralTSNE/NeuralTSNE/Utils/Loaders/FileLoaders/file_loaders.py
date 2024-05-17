from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from NeuralTSNE.DatasetLoader import get_datasets
from NeuralTSNE.Utils.Writers.StatWriters import (
    save_means_and_vars,
)
from NeuralTSNE.Utils.Writers.LabelWriters import (
    save_torch_labels,
)
from NeuralTSNE.Utils.Preprocessing import prepare_data


def load_text_file(
    input_file: str,
    step: int,
    header: bool,
    exclude_cols: List[int],
    variance_threshold: float,
) -> torch.Tensor:
    """
    Load and preprocess data from a text file.

    The function reads the data from the specified text file, skips the `header` if present,
    and excludes specified columns if the `exclude_cols` list is provided. It then subsamples
    the data based on the given `step` size. Finally, it preprocesses the data by applying
    a `variance threshold` to perform feature selection and returns the resulting `torch.Tensor`.

    Parameters
    ----------
    `input_file` : `str`
        The path to the input text file.
    `step` : `int`
        Step size for subsampling the data.
    `header` : `bool`
        A boolean indicating whether the file has a header.
    `exclude_cols` : `List[int]`
        A list of column indices to exclude from the data.
    `variance_threshold` : `float`
        Threshold for variance-based feature selection.

    Returns
    -------
    `torch.Tensor`
        Processed data tensor.
    """
    input_file = open(input_file, "r")
    cols = None
    if header:
        input_file.readline()
    if exclude_cols:
        last_pos = input_file.tell()
        ncols = len(input_file.readline().strip().split())
        input_file.seek(last_pos)
        cols = np.arange(0, ncols, 1)
        cols = tuple(np.delete(cols, exclude_cols))

    X = np.loadtxt(input_file, usecols=cols)

    input_file.close()

    data = np.array(X[::step, :])
    data = prepare_data(variance_threshold, data)

    return data


def load_npy_file(
    input_file: str,
    step: int,
    exclude_cols: List[int],
    variance_threshold: float,
) -> torch.Tensor:
    """
    Load and preprocess data from a `NumPy` (`.npy`) file.

    The function loads data from the specified `NumPy` file, subsamples it based on the given `step` size,
    and excludes specified columns if the `exclude_cols` list is provided. It then preprocesses the data
    by applying a `variance threshold` to perform feature selection and returns the resulting `torch.Tensor`.

    Parameters
    ----------
    `input_file` : `str`
        The path to the input `NumPy` file (`.npy`).
    `step` : `int`
        Step size for subsampling the data.
    `exclude_cols` : `List[int]`
        A list of column indices to exclude from the data.
    `variance_threshold` : `float`
        Threshold for variance-based feature selection.

    Returns
    -------
    `torch.Tensor`
        Processed data tensor.
    """
    data = np.load(input_file)
    data = data[::step, :]
    if exclude_cols:
        data = np.delete(data, exclude_cols, axis=1)

    data = prepare_data(variance_threshold, data)

    return data


def load_torch_dataset(name: str, step: int, output: str) -> Tuple[Dataset, Dataset]:
    """
    Load and preprocess a `torch.Dataset`, returning `training` and `testing` subsets.

    The function loads a `torch.Dataset` specified by the `name` parameter, extracts `training` and `testing` subsets,
    and preprocesses the `training` subset by saving labels and calculating means and variances.

    Parameters
    ----------
    `name` : `str`
        The name of the torch dataset to be loaded.
    `step` : `int`
        The step size for subsampling the training dataset.
    `output` : `str`
        The output file path for saving labels.

    Returns
    -------
    `Tuple[Dataset, Dataset]`
        A tuple containing the training and testing subsets.

    Note
    ----
    - The function uses the `name` parameter to load a torch dataset and extract training and testing subsets.
    - The training subset is subsampled using the `step` parameter.
    - Labels for the testing subset are saved to a file specified by the `output` parameter.
    - Means and variances for the training subset are calculated and saved to the `"means_and_vars.txt"` file.
    - The function returns a `tuple` containing the training and testing subsets.
    """
    train, test = get_datasets.get_dataset(name)
    train = Subset(train, range(0, len(train), step))

    save_torch_labels(output, test)
    train_data = torch.stack([row[0] for row in train])
    save_means_and_vars(train_data)

    return train, test
