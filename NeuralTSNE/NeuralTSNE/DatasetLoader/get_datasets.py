import os
from typing import List, Tuple

import torch
import numpy as np
from torch import flatten
from torch.utils.data import Dataset, TensorDataset, random_split
from torchvision import datasets
from torchvision.transforms import Compose, Lambda, ToTensor


def get_mnist() -> Tuple[Dataset, Dataset]:
    """
    Retrieves the MNIST dataset from `torchvision`.

    Returns
    -------
    `Tuple[Dataset, Dataset]`
        Tuple containing training and testing datasets.
    """
    mnist_dataset_train = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=Compose([ToTensor(), Lambda(flatten)]),
    )

    mnist_dataset_test = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=Compose([ToTensor(), Lambda(flatten)]),
    )
    return mnist_dataset_train, mnist_dataset_test


def get_fashion_mnist() -> Tuple[Dataset, Dataset]:
    """
    Retrieves the Fashion MNIST dataset from `torchvision`.

    Returns
    -------
    `Tuple[Dataset, Dataset]`
        Tuple containing training and testing datasets.
    """
    fashion_mnist_dataset_train = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=Compose([ToTensor(), Lambda(flatten)]),
    )

    fashion_mnist_dataset_test = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=Compose([ToTensor(), Lambda(flatten)]),
    )

    return fashion_mnist_dataset_train, fashion_mnist_dataset_test


def get_colvar() -> Tuple[Dataset, Dataset]:
    """
    Loads the 'colvar.csv' data and returns a train and test TensorDataset.

    Returns
    -------
    `Tuple[Dataset, Dataset]`
        Tuple containing training and testing datasets.
    """
    with open("colvar.csv", "r") as input_file:
        exclude_cols = [0, 1, 2, 3, 4]  # Exclude the first 5 columns
        cols = None

        input_file.readline()  # Ignore the header line

        # Remove excluded columns from the data
        last_pos = input_file.tell()
        ncols = len(input_file.readline().strip().split())
        input_file.seek(last_pos)
        cols = np.arange(0, ncols, 1)
        cols = tuple(np.delete(cols, exclude_cols))

        # Load the data
        data = np.loadtxt(input_file, usecols=cols)

    X = data
    X_torch = torch.tensor(X, dtype=torch.float32)

    dataset = TensorDataset(X_torch)

    # Split into training and testing sets
    total_size = len(dataset)
    test_size = int(0.2 * total_size)
    train_size = total_size - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    return train_dataset, test_dataset


def _get_available_datasets() -> List[str]:
    """
    Gets list of available datasets.

    Returns
    -------
    `List[str]`
        List of available datasets.
    """
    methods = [key[4:] for key in globals().keys() if key.startswith("get")]
    methods.remove("dataset")
    return methods


def prepare_dataset(dataset_name: str) -> Tuple[Dataset, Dataset]:
    """
    Loads the dataset from file or creates it if it does not exist.
    Returns the training and testing datasets.

    Parameters
    ----------
    `dataset_name` : `str`
        Name of the dataset.

    Returns
    -------
    `Tuple[Dataset, Dataset]`
        Tuple containing training and testing datasets.
    """
    if not (
        os.path.exists(dataset_name + "_train.data")
        and os.path.exists(dataset_name + "_test.data")
    ):
        train, test = globals()["get_" + dataset_name]()
        torch.save(train, dataset_name + "_train.data")
        torch.save(test, dataset_name + "_test.data")
    else:
        train = torch.load(dataset_name + "_train.data", weights_only=False)
        test = torch.load(dataset_name + "_test.data", weights_only=False)
    return train, test


def get_dataset(dataset_name: str) -> Tuple[Dataset, Dataset] | Tuple[None, None]:
    """
    Gets the dataset from the available datasets.

    Parameters
    ----------
    `dataset_name` : `str`
        Name of the dataset.

    Returns
    -------
    `Tuple[Dataset, Dataset]` | `Tuple[None, None]`
        Tuple containing training and testing datasets
        or None if the dataset is not available.
    """
    name = dataset_name.lower()
    if name in _get_available_datasets():
        return prepare_dataset(name)
    return None, None
