import argparse
from collections import OrderedDict
import io

import numpy as np
import lightning as L
from lightning.pytorch.tuner import Tuner
import torch
import torch.nn as nn
import torch.optim as optim

# from argparse_range import range_action
from lightning.pytorch.callbacks import EarlyStopping
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split, Subset
from tqdm import tqdm
import sys
from NeuralTSNE.DatasetLoader import get_datasets
from typing import Any, Callable, List, Tuple, Union
import torchinfo


def does_sum_up_to(a: float, b: float, to: float, epsilon=1e-7) -> bool:
    """
    Check if the sum of two numbers, `a` and `b`, is approximately equal to a target value `to` within a given `epsilon`.

    ---
    ### Parameters:
        - `a` (`float`): The first number.
        - `b` (`float`): The second number.
        - `to` (`float`): The target sum value.
        - `epsilon` (`float`, optional): The acceptable margin of error. Defaults to `1e-7`.

    ---
    ### Returns:
        - `bool`: `True` if the sum of `a` and `b` is within `epsilon` of the target value `to`, `False` otherwise.
    """
    return abs(a + b - to) < epsilon


def normalize_columns(data: torch.Tensor) -> torch.Tensor:
    """Normalize the columns of a 2D `torch.Tensor` to have values in the range `[0, 1]`.

    ---
    ### Parameters:
        - `data` (`torch.Tensor`): The input 2D tensor with columns to be normalized.

    ---
    ### Returns:
        - `torch.Tensor`: A new tensor with columns normalized to the range `[0, 1]`.

    ---
    ### Note:
    The normalization is done independently for each column, ensuring that the values in each column are scaled to the range `[0, 1]`.
    """
    data_min = data.min(dim=0)[0]
    data_range = data.max(dim=0)[0] - data_min
    return (data - data_min) / data_range


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
    vars = data.var(axis=0)
    cols = np.where(vars > variance_threshold)[0]
    filtered_data = data[:, cols]
    return filtered_data


def save_means_and_vars(data: torch.Tensor, filtered_data: torch.Tensor = None) -> None:
    """
    Calculate and save the means and variances of columns in a 2D `torch.Tensor` to a file.

    If `filtered_data` is provided, it calculates and saves means and variances for both original and filtered columns.

    ---
    ### Parameters:
        - `data` (`torch.Tensor`): The input 2D tensor for which means and variances are calculated.
        - `filtered_data` (`torch.Tensor`, optional): A filtered version of the input data. Defaults to `None`.

    ---
    ### Note:
    - The function calculates means and variances for each column in the input data.
    - If `filtered_data` is provided, it also calculates and saves means and variances for the corresponding filtered columns.
    - The results are saved to a `file` named `"means_and_vars.txt"` in a tabular format.
    """
    means = data.mean(axis=0)
    vars = data.var(axis=0)

    if filtered_data is not None:
        filtered_means = filtered_data.mean(axis=0)
        filtered_vars = filtered_data.var(axis=0)

    with open("means_and_vars.txt", "w") as f:
        f.writelines("column\tmean\tvar\n")
        for v in range(len(means)):
            f.writelines(f"{v}\t{means[v]}\t{vars[v]}\n")
        if filtered_data is not None:
            f.writelines("\nfiltered_column\tfiltered_mean\tfiltered_var\n")
            for v in range(len(filtered_means)):
                f.writelines(f"{v}\t{filtered_means[v]}\t{filtered_vars[v]}\n")


def load_torch_dataset(name: str, step: int, output: str) -> Tuple[Dataset, Dataset]:
    """
    Load and preprocess a `torch.Dataset`, returning `training` and `testing` subsets.

    The function loads a `torch.Dataset` specified by the `name` parameter, extracts `training` and `testing` subsets,
    and preprocesses the `training` subset by saving labels and calculating means and variances.

    ---
    ### Parameters:
        - `name` (`str`): The name of the torch dataset to be loaded.
        - `step` (`int`): The step size for subsampling the training dataset.
        - `output` (`str`): The output file path for saving labels.

    ---
    ### Returns:
        - `Tuple[Dataset, Dataset]`: A tuple containing the training and testing subsets.

    ---
    ### Note:
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


def save_torch_labels(output: str, test: Dataset) -> None:
    """
    Save labels from a `torch.Dataset` to a text file.

    The function extracts labels from the provided `test` dataset and saves them to a text file.
    The output file is named based on the provided `output` parameter.

    ---
    ### Parameters:
        - `output` (`str`): The output file path for saving labels.
        - `test` (`Dataset`): The `torch.Dataset` containing labels to be saved.

    ---
    ### Note:
    - The function iterates through the `test` dataset, extracts labels, and saves them to a text file.
    - The output file is named by appending `"_labels.txt"` to the `output` parameter, removing the file extension if present.
    """
    with open(
        output.rsplit(".", maxsplit=1)[0] + "_labels.txt",
        "w",
    ) as f:
        for _, row in tqdm(
            enumerate(test), unit="samples", total=len(test), desc="Saving labels"
        ):
            f.writelines(f"{row[1]}\n")


def load_labels(labels: io.TextIOWrapper) -> Union[torch.Tensor, None]:
    """
    Load labels from a text file into a `torch.Tensor`.

    The function reads labels from the provided text file and converts them into a `torch.Tensor` of type `float`.
    If the `labels` parameter is not provided or the file is empty, the function returns `None`.

    ---
    ### Parameters:
        - `labels` (`io.TextIOWrapper`): The `file` object containing labels to be loaded.

    ---
    ### Returns:
        - `Union[torch.Tensor, None]`: A `torch.Tensor` containing loaded labels or `None` if no labels are available.

    ---
    ### Note:
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

    ---
    ### Parameters:
        - `input_file` (`str`): The path to the input text file.
        - `step` (`int`): Step size for subsampling the data.
        - `header` (`bool`): A boolean indicating whether the file has a header.
        - `exclude_cols` (`List[int]`): A list of column indices to exclude from the data.
        - `variance_threshold` (`float`): Threshold for variance-based feature selection.

    ---
    ### Returns:
        - `torch.Tensor`: Processed data tensor.
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

    ---
    ### Parameters:
        - `input_file` (`str`): The path to the input `NumPy` file (`.npy`).
        - `step` (`int`): Step size for subsampling the data.
        - `exclude_cols` (`List[int]`): A list of column indices to exclude from the data.
        - `variance_threshold` (`float`): Threshold for variance-based feature selection.

    ---
    ### Returns:
        - `torch.Tensor`: Processed data tensor.
    """
    data = np.load(input_file)
    data = data[::step, :]
    if exclude_cols:
        data = np.delete(data, exclude_cols, axis=1)

    data = prepare_data(variance_threshold, data)

    return data


def prepare_data(variance_threshold: float, data: np.ndarray) -> torch.Tensor:
    """
    Prepare data for further analysis by filtering based on variance,
    saving means and variances, and normalizing columns.

    ---
    ### Parameters:
        - `variance_threshold` (`float`): Threshold for variance-based feature selection.
        - `data` (`np.ndarray`): Input data array.

    ---
    ### Returns:
        - `torch.Tensor`: Processed and normalized data tensor.

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


def Hbeta(D: torch.Tensor, beta: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate entropy and probability distribution based on a distance matrix.

    ---
    ### Parameters:
        - `D` (`torch.Tensor`): Distance matrix.
        - `beta` (`float`): Parameter for the computation.

    ---
    ### Returns:
        - `Tuple[torch.Tensor, torch.Tensor]`: Entropy and probability distribution.

    The function calculates the entropy and probability distribution based on
    the provided distance matrix (`D`) and the specified parameter (`beta`).
    """
    P = torch.exp(-D * beta)
    sumP = torch.sum(P)
    H = torch.log(sumP) + beta * torch.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p_job(
    data: Tuple[int, torch.Tensor, torch.Tensor],
    tolerance: float,
    max_iterations: int = 50,
) -> Tuple[int, torch.Tensor, torch.Tensor, int]:
    """
    Perform a binary search to find an appropriate value of `beta` for a given point.

    ---
    ### Parameters:
        - `data` (`Tuple[int, torch.Tensor, torch.Tensor]`): Tuple containing index, distance matrix, and target entropy.
        - `tolerance` (`float`): Tolerance level for convergence.
        - `max_iterations` (`int`, optional): Maximum number of iterations for the binary search. Default is 50.

    ---
    ### Returns:
        - `Tuple[int, torch.Tensor]`: Index, probability distribution, entropy difference, and number of iterations.

    The function performs a binary search to find an appropriate value of `beta` for a given point,
    aiming to match the target entropy. It returns the index, probability distribution, entropy difference, and number of iterations.
    """
    i, Di, logU = data
    beta = 1.0
    beta_min = -torch.inf
    beta_max = torch.inf

    H, thisP = Hbeta(Di, beta)
    Hdiff = H - logU

    it = 0
    while it < max_iterations and torch.abs(Hdiff) > tolerance:
        if Hdiff > 0:
            beta_min = beta
            if torch.isinf(torch.tensor(beta_max)):
                beta *= 2
            else:
                beta = (beta + beta_max) / 2
        else:
            beta_max = beta
            if torch.isinf(torch.tensor(beta_min)):
                beta /= 2
            else:
                beta = (beta + beta_min) / 2

        H, thisP = Hbeta(Di, beta)
        Hdiff = H - logU
        it += 1
    return i, thisP, Hdiff, it


def x2p(
    X: torch.Tensor,
    perplexity: int,
    tolerance: float,
) -> torch.Tensor:
    """
    Compute conditional probabilities using the t-SNE algorithm.

    ---
    ### Parameters:
        - `X` (`torch.Tensor`): Input data tensor.
        - `perplexity` (`int`): Perplexity parameter for t-SNE.
        - `tolerance` (`float`): Tolerance level for convergence.

    ---
    ### Returns:
        - `torch.Tensor`: Conditional probability matrix.
    """
    n = X.shape[0]
    logU = torch.log(torch.tensor([perplexity], device=X.device))

    sum_X = torch.sum(torch.square(X), dim=1)
    D = torch.add(torch.add(-2 * torch.mm(X, X.mT), sum_X).T, sum_X)

    idx = (1 - torch.eye(n)).type(torch.bool)
    D = D[idx].reshape((n, -1))

    P = torch.zeros(n, n, device=X.device)

    for i in range(n):
        P[i, idx[i]] = x2p_job((i, D[i], logU), tolerance)[1]
    return P


class NeuralNetwork(nn.Module):
    """
    Neural network model for dimensionality reduction.

    ---
    ### Parameters:
        - `initial_features` (`int`): Number of input features.
        - `n_components` (`int`): Number of components in the output.
        - `multipliers` (`List[float]`): List of multipliers for hidden layers.

    The neural network is designed for dimensionality reduction with
    hidden layers defined by the list of multipliers. ReLU activation
    functions are applied between layers.
    """

    def __init__(
        self, initial_features: int, n_components: int, multipliers: List[float]
    ) -> None:
        super(NeuralNetwork, self).__init__()
        layers = OrderedDict()
        layers["0"] = nn.Linear(
            initial_features, int(multipliers[0] * initial_features)
        )
        for i in range(1, len(multipliers)):
            layers["ReLu" + str(i - 1)] = nn.ReLU()
            layers[str(i)] = nn.Linear(
                int(multipliers[i - 1] * initial_features),
                int(multipliers[i] * initial_features),
            )
            layers["ReLu" + str(i)] = nn.ReLU()
        if len(multipliers) == 1:
            layers["ReLu" + str(len(multipliers) - 1)] = nn.ReLU()
        layers[str(len(multipliers))] = nn.Linear(
            int(multipliers[-1] * initial_features), n_components
        )
        self.linear_relu_stack = nn.Sequential(layers)

    def forward(self, x):
        """
        Forward pass through the neural network.

        ---
        ### Parameters:
            - `x` (`torch.Tensor`): Input tensor.

        ---
        ### Returns:
            - `torch.Tensor`: Output tensor.
        """
        logits = self.linear_relu_stack(x)
        return logits


class ParametricTSNE:
    """
    Parametric t-SNE implementation using a neural network model.

    ---
    ### Parameters:
        - `loss_fn` (`str`): Loss function for t-SNE. Currently supports `kl_divergence`.
        - `n_components` (`int`): Number of components in the output.
        - `perplexity` (`int`): Perplexity parameter for t-SNE.
        - `batch_size` (`int`): Batch size for training.
        - `early_exaggeration_epochs` (`int`): Number of epochs for early exaggeration.
        - `early_exaggeration_value` (`float`): Early exaggeration factor.
        - `max_iterations` (`int`): Maximum number of iterations for optimization.
        - `features` (`int`): Number of input features.
        - `multipliers` (`List[float]`): List of multipliers for hidden layers in the neural network.
        - `n_jobs` (`int`, optional): Number of workers for data loading. Default is `0`.
        - `tolerance` (`float`, optional): Tolerance level for convergence. Default is `1e-5`.
        - `force_cpu` (`bool`, optional): Force using CPU even if GPU is available. Default is `False`.
    """

    def __init__(
        self,
        loss_fn: str,
        n_components: int,
        perplexity: int,
        batch_size: int,
        early_exaggeration_epochs: int,
        early_exaggeration_value: float,
        max_iterations: int,
        features: int,
        multipliers: List[float],
        n_jobs: int = 0,
        tolerance: float = 1e-5,
        force_cpu: bool = False,
    ):
        if force_cpu or not torch.cuda.is_available():
            self.device = torch.device("cpu")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        self.model = NeuralNetwork(features, n_components, multipliers).to(self.device)
        torchinfo.summary(
            self.model,
            input_size=(batch_size, 1, features),
            col_names=(
                "input_size",
                "output_size",
                "num_params",
                "kernel_size",
                "mult_adds",
            ),
        )

        self.perplexity = perplexity
        self.batch_size = batch_size
        self.early_exaggeration_epochs = early_exaggeration_epochs
        self.early_exaggeration_value = early_exaggeration_value
        self.n_jobs = n_jobs
        self.tolerance = tolerance
        self.max_iterations = max_iterations

        self.loss_fn = self.set_loss_fn(loss_fn)

    def set_loss_fn(self, loss_fn) -> Callable:
        """
        Set the loss function based on the provided string.

        ---
        ### Parameters:
            - `loss_fn` (`str`): String indicating the desired loss function.

        ---
        ### Returns:
            - `Callable`: Corresponding loss function.

        Currently supports `kl_divergence` as the loss function.
        """
        fn = None
        if loss_fn == "kl_divergence":
            fn = self._kl_divergence
        self.loss_fn = fn
        return fn

    def save_model(self, filename: str):
        """
        Save the model's state dictionary to a file.

        ---
        ### Parameters:
            - `filename` (`str`): Name of the file to save the model.
        """
        torch.save(self.model.state_dict(), filename)

    def read_model(self, filename: str):
        """
        Load the model's state dictionary from a file.

        ---
        ### Parameters:
            - `filename` (`str`): Name of the file to load the model.
        """
        self.model.load_state_dict(torch.load(filename))

    def split_dataset(
        self,
        X: torch.Tensor,
        y: torch.Tensor = None,
        train_size: float = None,
        test_size: float = None,
    ) -> Tuple[Union[DataLoader, None], Union[DataLoader, None]]:
        """
        Split the dataset into training and testing sets.

        ---
        ### Parameters:
            - `X` (`torch.Tensor`): Input data tensor.
            - `y` (`torch.Tensor`, optional): Target tensor. Default is `None`.
            - `train_size` (`float`, optional): Proportion of the dataset to include in the training set.
            - `test_size` (`float`, optional): Proportion of the dataset to include in the testing set.

        ---
        ### Returns:
            - `Tuple[Union[DataLoader, None], Union[DataLoader, None]]`: Tuple containing training and testing dataloaders.

        Splits the input data into training and testing sets, and returns corresponding dataloaders.
        """
        train_size, test_size = self._determine_train_test_split(train_size, test_size)
        if y is None:
            dataset = TensorDataset(X)
        else:
            dataset = TensorDataset(X, y)
        train_size = int(train_size * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        if train_size == 0:
            train_dataset = None
        if test_size == 0:
            test_dataset = None

        return self.create_dataloaders(train_dataset, test_dataset)

    def _determine_train_test_split(
        self, train_size: float, test_size: float
    ) -> Tuple[float, float]:
        """
        Determine the proportions of training and testing sets.

        ---
        ### Parameters:
            - `train_size` (`float`): Proportion of the dataset to include in the training set.
            - `test_size` (`float`): Proportion of the dataset to include in the testing set.

        ---
        ### Returns:
            - `Tuple[float, float]`: Tuple containing the determined proportions.
        """
        if train_size is None and test_size is None:
            train_size = 0.8
            test_size = 1 - train_size
        elif train_size is None:
            train_size = 1 - test_size
        elif test_size is None:
            test_size = 1 - train_size
        elif not does_sum_up_to(train_size, test_size, 1):
            test_size = 1 - train_size
        return train_size, test_size

    def create_dataloaders(
        self, train: Dataset, test: Dataset
    ) -> Tuple[Union[DataLoader, None], Union[DataLoader, None]]:
        """
        Create dataloaders for training and testing sets.

        ---
        ### Parameters:
            - `train` (`Dataset`): Training dataset.
            - `test` (`Dataset`): Testing dataset.

        ---
        ### Returns:
            - `Tuple[Union[DataLoader, None], Union[DataLoader, None]]`: Tuple containing training and testing dataloaders.
        """
        train_loader = (
            DataLoader(
                train,
                batch_size=self.batch_size,
                drop_last=True,
                pin_memory=False if self.device == "cpu" else True,
                num_workers=self.n_jobs if self.device == "cpu" else 0,
            )
            if train is not None
            else None
        )
        test_loader = (
            DataLoader(
                test,
                batch_size=self.batch_size,
                drop_last=False,
                pin_memory=False if self.device == "cpu" else True,
                num_workers=self.n_jobs if self.device == "cpu" else 0,
            )
            if test is not None
            else None
        )
        return train_loader, test_loader

    def _calculate_P(self, dataloader: DataLoader) -> torch.Tensor:
        """
        Calculate joint probability matrix P.

        ---
        ### Parameters:
            - `dataloader` (`DataLoader`): Dataloader for the dataset.

        ---
        ### Returns:
            - `torch.Tensor`: Conditional probability matrix P.
        """
        n = len(dataloader.dataset)
        P = torch.zeros((n, self.batch_size), device=self.device)
        for i, (X, *_) in tqdm(
            enumerate(dataloader),
            unit="batch",
            total=len(dataloader),
            desc="Calculating P",
        ):
            batch = x2p(X, self.perplexity, self.tolerance)
            batch[torch.isnan(batch)] = 0
            batch = batch + batch.mT
            batch = batch / batch.sum()
            batch = torch.maximum(
                batch.to(self.device), torch.tensor([1e-12], device=self.device)
            )
            P[i * self.batch_size : (i + 1) * self.batch_size] = batch
        return P

    def _kl_divergence(self, Y: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        """
        Calculate the Kullback-Leibler divergence.

        ---
        ### Parameters:
            - `Y` (`torch.Tensor`): Embedding tensor.
            - `P` (`torch.Tensor`): Conditional probability matrix.

        ---
        ### Returns:
            - `torch.Tensor`: Kullback-Leibler divergence.

        Calculates the Kullback-Leibler divergence between the true conditional probability matrix P
        and the conditional probability matrix Q based on the current embedding Y.
        """
        sum_Y = torch.sum(torch.square(Y), dim=1)
        eps = torch.tensor([1e-15], device=self.device)
        D = (
            sum_Y + torch.reshape(sum_Y, [-1, 1]) - 2 * torch.matmul(Y, Y.mT)
        )  # TODO: cdist may be the way to calculate distances neatly
        Q = torch.pow(1 + D / 1.0, -(1.0 + 1) / 2)
        Q *= 1 - torch.eye(self.batch_size, device=self.device)
        Q /= torch.sum(Q)
        Q = torch.maximum(Q, eps)
        C = torch.log((P + eps) / (Q + eps))
        C = torch.sum(P * C)
        return C


class Classifier(L.LightningModule):
    """
    Lightning Module for training a classifier using a Parametric t-SNE model.

    ---
    ### Parameters:
        - `tsne` (`ParametricTSNE`): Parametric t-SNE model for feature extraction.
        - `shuffle` (`bool`): Flag indicating whether to shuffle data during training.
        - `optimizer` (`str`, optional): Optimizer for training. Default is `adam`.
        - `lr` (`float`, optional): Learning rate for the optimizer. Default is `1e-3`.

    This class defines a Lightning Module for training a classifier using a Parametric t-SNE model
    for feature extraction. It includes methods for the training step, configuring optimizers, and
    handling the training process.
    """

    def __init__(
        self,
        tsne: ParametricTSNE,
        shuffle: bool,
        optimizer: str = "adam",
        lr: float = 1e-3,
    ):
        super().__init__()
        self.tsne = tsne
        self.batch_size = tsne.batch_size
        self.model = self.tsne.model
        self.loss_fn = tsne.loss_fn
        self.exaggeration_epochs = tsne.early_exaggeration_epochs
        self.exaggeration_value = tsne.early_exaggeration_value
        self.shuffle = shuffle
        self.lr = lr
        self.optimizer = optimizer
        self.reset_exaggeration_status()

    def reset_exaggeration_status(self):
        """
        Reset exaggeration status based on the number of exaggeration epochs.
        """
        self.has_exaggeration_ended = True if self.exaggeration_epochs == 0 else False

    def training_step(
        self,
        batch: Union[
            torch.Tensor, Tuple[torch.Tensor, ...], List[Union[torch.Tensor, Any]]
        ],
        batch_idx: int,
    ):
        """
        Perform a single training step.

        ---
        ### Parameters:
            - `batch` (`Union[torch.Tensor, Tuple[torch.Tensor, ...], List[Union[torch.Tensor, Any]]]`): Input batch.
            - `batch_idx` (`int`): Index of the current batch.

        ---
        ### Returns:
            - `Dict[str, torch.Tensor]`: Dictionary containing the `loss` value.

        This method defines a single training step for the classifier. It computes the loss using
        the model's logits and the conditional probability matrix _P_batch.
        """
        x = batch[0]
        _P_batch = self.P_current[
            batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size
        ]

        if self.shuffle:
            p_idxs = torch.randperm(x.shape[0])
            x = x[p_idxs]
            _P_batch = _P_batch[p_idxs, :]
            _P_batch = _P_batch[:, p_idxs]

        logits = self.model(x)
        loss = self.loss_fn(logits, _P_batch)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return {"loss": loss}

    def _set_optimizer(
        self, optimizer: str, optimizer_params: dict
    ) -> torch.optim.Optimizer:
        """
        Set the optimizer based on the provided string.

        ---
        ### Parameters:
            - `optimizer` (`str`): String indicating the desired optimizer.
            - `optimizer_params` (`dict`): Dictionary containing optimizer parameters.

        ---
        ### Returns:
            - `torch.optim.Optimizer`: Initialized optimizer.

        This method initializes and returns the desired optimizer based on the provided string.
        """
        if optimizer == "adam":
            return optim.Adam(self.model.parameters(), **optimizer_params)
        elif optimizer == "sgd":
            return optim.SGD(self.model.parameters(), **optimizer_params)
        elif optimizer == "rmsprop":
            return optim.RMSprop(self.model.parameters(), **optimizer_params)
        else:
            raise ValueError("Unknown optimizer")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure the optimizer for training.

        ---
        ### Returns:
            - `torch.optim.Optimizer`: Configured optimizer.

        This method configures and returns the optimizer for training based on the specified parameters.
        """
        return self._set_optimizer(self.optimizer, {"lr": self.lr})

    def on_train_start(self) -> None:
        """
        Perform actions at the beginning of the training process.

        This method is called at the start of the training process and calculates the conditional
        probability matrix P based on the training dataloader.
        """
        if not hasattr(self, "P"):
            self.P = self.tsne._calculate_P(self.trainer.train_dataloader)

    def on_train_epoch_start(self) -> None:
        """
        Perform actions at the start of each training epoch.

        This method is called at the start of each training epoch. If exaggeration is enabled and has
        not ended, it modifies the conditional probability matrix for the current epoch.
        """
        if self.current_epoch > 0 and self.has_exaggeration_ended:
            return
        if (
            self.exaggeration_epochs > 0
            and self.current_epoch < self.exaggeration_epochs
        ):
            if not hasattr(self, "P_multiplied"):
                self.P_multiplied = self.P.clone()
                self.P_multiplied *= self.exaggeration_value
            self.P_current = self.P_multiplied
        else:
            self.P_current = self.P
            self.has_exaggeration_ended = True

    def on_train_epoch_end(self) -> None:
        """
        Perform actions at the end of each training epoch.

        This method is called at the end of each training epoch. If exaggeration has ended and
        P_multiplied exists, it is deleted to free up memory.
        """
        if hasattr(self, "P_multiplied") and self.has_exaggeration_ended:
            del self.P_multiplied

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        """
        Perform a single step during the prediction process.

        ---
        ### Parameters:
            - `batch`: Input batch.
            - `batch_idx`: Index of the current batch.
            - `dataloader_idx`: Index of the dataloader (optional).

        ---
        ### Returns:
            - `torch.Tensor`: Model predictions.

        This method is called during the prediction process and returns the model's predictions for the input batch.
        """
        return self.model(batch[0])


class FileTypeWithExtensionCheck(argparse.FileType):
    """
    Custom `argparse.FileType` with additional extension validation.

    ---
    ### Parameters:
        - `mode` (`str`, optional): File mode. Default is `r`.
        - `valid_extensions` (`str` or `Tuple[str, ...]`, optional): Valid file extensions.
        - `**kwargs`: Additional keyword arguments.

    This class extends argparse.FileType to include validation of file extensions.
    """

    def __init__(self, mode="r", valid_extensions=None, **kwargs):
        super().__init__(mode, **kwargs)
        self.valid_extensions = valid_extensions

    def __call__(self, string):
        """
        Validate the file extension before calling the parent `__call__` method.

        ---
        ### Parameters:
            - `string` (`str`): Input string representing the filename.

        ---
        ### Returns:
            - `file`: File object.

        This method performs additional validation on the file extension before calling
        the parent `__call__` method from `argparse.FileType`.
        """
        if self.valid_extensions:
            if not string.endswith(self.valid_extensions):
                raise argparse.ArgumentTypeError("Not a valid filename extension!")
        return super().__call__(string)


class FileTypeWithExtensionCheckWithPredefinedDatasets(FileTypeWithExtensionCheck):
    """
    Custom `argparse.FileType` with additional extension and predefined dataset validation.

    ---
    ### Parameters:
        - `mode` (`str`, optional): File mode. Default is `r`.
        - `valid_extensions` (`str` or `Tuple[str, ...]`, optional): Valid file extensions.
        - `available_datasets` (`List[str]`, optional): List of available datasets.
        - `**kwargs`: Additional keyword arguments.

    This class extends `FileTypeWithExtensionCheck` to include validation of predefined datasets.
    """

    def __init__(
        self, mode="r", valid_extensions=None, available_datasets=None, **kwargs
    ):
        super().__init__(mode, valid_extensions, **kwargs)
        self.available_datasets = available_datasets or []

    def __call__(self, string):
        """
        Validate the file extension and predefined dataset before calling the parent `__call__` method.

        ---
        ### Parameters:
            - `string` (`str`): Input string representing the filename.

        ---
        ### Returns:
            - `file` or `str`: File object or predefined dataset name.

        This method performs additional validation on the file extension and predefined dataset before calling
        the parent `__call__` method from `FileTypeWithExtensionCheck`.
        """
        if len(self.available_datasets) > 0 and string in self.available_datasets:
            return string
        return super().__call__(string)


def save_results(args: dict, test: DataLoader, Y: Union[List[Any], List[List[Any]]]):
    """
    Save results to a file.

    ---
    ### Parameters:
        - `args` (`dict`): Dictionary containing arguments, including the output file path (`o`) and step size (`step`).
        - `test` (`DataLoader`): DataLoader for the test dataset.
        - `Y` (`Union[List[Any], List[List[Any]]]`): List of results to be saved.

    This function saves the results to a file specified by the output file path in the arguments.
    """
    if test is not None:
        with open(args["o"], "w") as f:
            f.writelines(f"{args['step']}\n")
            for _, batch in tqdm(
                enumerate(Y), unit="batches", total=(len(Y)), desc="Saving results"
            ):
                for entry in batch:
                    output_line = "\t".join([str(x) for x in entry])
                    f.writelines(f"{output_line}\n")


def save_labels_data(
    args: dict,
    test: DataLoader,
):
    """
    Save labels data to a new file.

    ---
    ### Parameters:
        - `args` (`dict`): Dictionary containing arguments, including the output file path (`o`).
        - `test` (`DataLoader`): DataLoader for the test dataset.

    This function saves the labels data to a new file with a name based on the original output file path.
    """
    if test is not None:
        new_name = args["o"].rsplit(".", 1)[0] + "_labels.txt"
        with open(new_name, "w") as f:
            for i, batch in tqdm(
                enumerate(test),
                unit="batches",
                total=(len(test)),
                desc="Saving new labels",
            ):
                for samples in batch:
                    samples = samples.tolist()
                    for sample in samples:
                        for col in sample:
                            f.write(str(col))
                            f.write("\t")
                        f.write("\n")


def run_tsne(
    input_file,
    iter=1000,
    labels=None,
    no_dims=2,
    perplexity=30.0,
    exclude_cols=None,
    step=1,
    exaggeration_iter=0,
    exaggeration_value=12,
    o="result.txt",
    model_save=None,
    model_load=None,
    shuffle=False,
    train_size=None,
    test_size=None,
    jobs=1,
    batch_size=1000,
    header=False,
    net_multipliers=None,
    variance_threshold=None,
    cpu=False,
    early_stopping_delta=1e-5,
    early_stopping_patience=3,
    lr=1e-3,
    auto_lr=False,
):
    available_datasets = []
    if "NeuralTSNE.DatasetLoader.get_datasets" in sys.modules:
        available_datasets = get_datasets._get_available_datasets()

    if net_multipliers is None:
        net_multipliers = [0.75, 0.75, 0.75]

    skip_data_splitting = False
    if (
        not isinstance(input_file, io.TextIOWrapper)
        and len(available_datasets) > 0
        and (name := input_file.lower()) in available_datasets
    ):
        train, test = load_torch_dataset(name, step, o)
        skip_data_splitting = True
        features = np.prod(train.dataset.data.shape[1:])
    else:
        labels = load_labels(labels)

        if input_file.endswith(".npy"):
            data = load_npy_file(input_file, step, exclude_cols, variance_threshold)
        else:
            data = load_text_file(
                input_file, step, header, exclude_cols, variance_threshold
            )
        features = data.shape[1]

    tsne = ParametricTSNE(
        loss_fn="kl_divergence",
        n_components=no_dims,
        perplexity=perplexity,
        batch_size=batch_size,
        early_exaggeration_epochs=exaggeration_iter,
        early_exaggeration_value=exaggeration_value,
        max_iterations=iter,
        features=features,
        multipliers=net_multipliers,
        n_jobs=jobs,
        force_cpu=cpu,
    )

    early_stopping = EarlyStopping(
        "train_loss_epoch",
        min_delta=early_stopping_delta,
        patience=early_stopping_patience,
    )

    is_gpu = tsne.device == torch.device("cuda:0")

    trainer = L.Trainer(
        accelerator="gpu" if is_gpu else "cpu",
        devices=1 if is_gpu else tsne.n_jobs,
        log_every_n_steps=1,
        max_epochs=tsne.max_iterations,
        callbacks=[early_stopping],
    )

    classifier = Classifier(tsne, shuffle, lr=lr)

    if model_load:
        tsne.read_model(model_load)
        train, test = (
            tsne.split_dataset(data, y=labels, test_size=1)
            if not skip_data_splitting
            else tsne.create_dataloaders(train, test)
        )
        if not skip_data_splitting:
            save_labels_data({"o": o}, test)
        Y = trainer.predict(classifier, test)
    else:
        train, test = (
            tsne.split_dataset(
                data, y=labels, train_size=train_size, test_size=test_size
            )
            if not skip_data_splitting
            else tsne.create_dataloaders(train, test)
        )
        if auto_lr:
            tuner = Tuner(trainer)
            tuner.lr_find(classifier, train)
            classifier.reset_exaggeration_status()
        if not skip_data_splitting:
            save_labels_data({"o": o}, test)
        trainer.fit(classifier, train)
        if model_save:
            tsne.save_model(model_save)
        if test is not None:
            Y = trainer.predict(classifier, test)

    save_results({"o": o, "step": step}, test, Y)


# if __name__ == "__main__":
# run_tsne("dialanine.npy", 10, no_dims=2, perplexity=30)
# available_datasets = []
# if "NeuralTSNE.DatasetLoader.get_datasets" in sys.modules:
#     available_datasets = get_datasets._get_available_datasets()

# parser = argparse.ArgumentParser(description="t-SNE Algorithm")
# parser.add_argument(
#     "input_file",
#     type=FileTypeWithExtensionCheckWithPredefinedDatasets(
#         valid_extensions=("txt", "data", "npy"),
#         available_datasets=available_datasets,
#     ),
#     help="Input file",
# )
# parser.add_argument(
#     "-iter", type=int, default=1000, help="Number of iterations", required=False
# )
# parser.add_argument(
#     "-labels",
#     type=FileTypeWithExtensionCheck(valid_extensions=("txt", "data")),
#     help="Labels file",
#     required=False,
# )
# parser.add_argument(
#     "-no_dims", type=int, help="Number of dimensions", required=True, default=2
# )
# parser.add_argument(
#     "-perplexity",
#     type=float,
#     help="Perplexity of the Gaussian kernel",
#     required=True,
#     default=30.0,
# )
# parser.add_argument(
#     "-exclude_cols", type=int, nargs="+", help="Columns to exclude", required=False
# )
# parser.add_argument(
#     "-step", type=int, help="Step between samples", required=False, default=1
# )
# parser.add_argument(
#     "-exaggeration_iter",
#     type=int,
#     help="Early exaggeration end",
#     required=False,
#     default=0,
# )
# parser.add_argument(
#     "-exaggeration_value",
#     type=float,
#     help="Early exaggeration value",
#     required=False,
#     default=12,
# )
# parser.add_argument(
#     "-o", type=str, help="Output filename", required=False, default="result.txt"
# )
# parser.add_argument(
#     "-model_save",
#     type=str,
#     help="Model save filename",
#     required=False,
# )
# parser.add_argument(
#     "-model_load",
#     type=str,
#     help="Model filename to load",
#     required=False,
# )
# parser.add_argument("-shuffle", action="store_true", help="Shuffle data")
# parser.add_argument(
#     "-train_size",
#     type=float,
#     action=range_action(0, 1),
#     help="Train size",
#     required=False,
# )
# parser.add_argument(
#     "-test_size",
#     type=float,
#     action=range_action(0, 1),
#     help="Test size",
#     required=False,
# )
# parser.add_argument(
#     "-jobs", type=int, help="Number of jobs", required=False, default=1
# )
# parser.add_argument(
#     "-batch_size", type=int, help="Batch size", required=False, default=1000
# )

# parser.add_argument("-header", action="store_true", help="Data has header")
# parser.add_argument(
#     "-net_multipliers",
#     type=float,
#     nargs="+",
#     help="Network multipliers",
#     default=[0.75, 0.75, 0.75],
# )
# parser.add_argument("-variance_threshold", type=float, help="Variance threshold")
# parser.add_argument("-cpu", action="store_true", help="Use CPU")
# parser.add_argument(
#     "-early_stopping_delta", type=float, help="Early stopping delta", default=1e-5
# )
# parser.add_argument(
#     "-early_stopping_patience", type=int, help="Early stopping patience", default=3
# )
# parser.add_argument("-lr", type=float, help="Learning rate", default=1e-3)
# parser.add_argument("-auto_lr", action="store_true", help="Auto learning rate")

# args = parser.parse_args(
#     ["dialanine.npy", "-no_dims", "2", "-perplexity", "30", "-iter", "10"]
# )

# args = parser.parse_args()

# skip_data_splitting = False
# if (
#     not isinstance(args.input_file, io.TextIOWrapper)
#     and len(available_datasets) > 0
#     and (name := args.input_file.lower()) in available_datasets
# ):
#     train, test = load_torch_dataset(name, args.step, args.o)
#     skip_data_splitting = True
#     features = np.prod(train.dataset.data.shape[1:])
# else:
#     labels = load_labels(args.labels)

#     if args.input_file.name.endswith(".npy"):
#         data = load_npy_file(
#             args.input_file, args.step, args.exclude_cols, args.variance_threshold
#         )
#     else:
#         data = load_text_file(
#             args.input_file,
#             args.step,
#             args.header,
#             args.exclude_cols,
#             args.variance_threshold,
#         )
#     features = data.shape[1]

# tsne = ParametricTSNE(
#     loss_fn="kl_divergence",
#     n_components=args.no_dims,
#     perplexity=args.perplexity,
#     batch_size=args.batch_size,
#     early_exaggeration_epochs=args.exaggeration_iter,
#     early_exaggeration_value=args.exaggeration_value,
#     max_iterations=args.iter,
#     features=features,
#     multipliers=args.net_multipliers,
#     n_jobs=args.jobs,
#     force_cpu=args.cpu,
# )

# early_stopping = EarlyStopping(
#     "train_loss_epoch",
#     min_delta=args.early_stopping_delta,
#     patience=args.early_stopping_patience,
# )

# is_gpu = tsne.device == torch.device("cuda:0")
# trainer = L.Trainer(
#     accelerator="gpu" if is_gpu else "cpu",
#     devices=1 if is_gpu else tsne.n_jobs,
#     log_every_n_steps=1,
#     max_epochs=tsne.max_iterations,
#     callbacks=[early_stopping],
# )

# classifier = Classifier(tsne, args.shuffle, lr=args.lr)

# if args.model_load:
#     tsne.read_model(args.model_load)
#     train, test = (
#         tsne.split_dataset(data, y=labels, test_size=1)
#         if not skip_data_splitting
#         else tsne.create_dataloaders(train, test)
#     )
#     if not skip_data_splitting:
#         save_labels_data(args, test)
#     Y = trainer.predict(classifier, test)
# else:
#     train, test = (
#         tsne.split_dataset(
#             data, y=labels, train_size=args.train_size, test_size=args.test_size
#         )
#         if not skip_data_splitting
#         else tsne.create_dataloaders(train, test)
#     )
#     if args.auto_lr:
#         tuner = Tuner(trainer)
#         tuner.lr_find(tsne.model, train)
#     if not skip_data_splitting:
#         save_labels_data(args, test)
#     trainer.fit(classifier, train)
#     if args.model_save:
#         tsne.save_model(args.model_save)
#     if test is not None:
#         Y = trainer.predict(classifier, test)

# save_results(args, test, Y)
