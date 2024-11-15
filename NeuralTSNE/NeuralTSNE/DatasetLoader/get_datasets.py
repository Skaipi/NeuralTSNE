import os
from typing import List, Tuple

import torch
from torch import flatten
from torch.utils.data import Dataset
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
        train = torch.load(dataset_name + "_train.data")
        test = torch.load(dataset_name + "_test.data")
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
