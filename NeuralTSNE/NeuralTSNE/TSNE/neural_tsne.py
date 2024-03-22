import io
import sys
from collections import OrderedDict
from typing import Any, Callable, List, Tuple, Union

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchinfo
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.tuner import Tuner
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from tqdm import tqdm

from NeuralTSNE.DatasetLoader import get_datasets
from NeuralTSNE.Utils.Writers.StatWriters.stat_writers import save_results
from NeuralTSNE.Utils.Writers.LabelWriters.label_writers import save_labels_data
from NeuralTSNE.Utils.Loaders.LabelLoaders.label_loaders import load_labels
from NeuralTSNE.Utils.Loaders.FileLoaders.file_loaders import (
    load_npy_file,
    load_text_file,
    load_torch_dataset,
)
from NeuralTSNE.Utils.utils import does_sum_up_to


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
