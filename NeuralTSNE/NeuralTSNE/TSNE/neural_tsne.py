import io
import sys
import os
from typing import Tuple
import argparse
from argparse_range import range_action

import numpy as np
import torch

import pytorch_lightning as L
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.tuner import Tuner

from NeuralTSNE.DatasetLoader import get_datasets
from NeuralTSNE.Utils.Validators.FileTypeValidators import (
    FileTypeWithExtensionCheck,
    FileTypeWithExtensionCheckWithPredefinedDatasets,
)
from NeuralTSNE.Utils.Writers.StatWriters import save_results
from NeuralTSNE.Utils.Writers.LabelWriters import save_labels_data
from NeuralTSNE.Utils.Loaders.LabelLoaders import load_labels
from NeuralTSNE.Utils.Loaders.FileLoaders import (
    load_npy_file,
    load_text_file,
    load_torch_dataset,
)

from NeuralTSNE.TSNE.ParametricTSNE import ParametricTSNE
from NeuralTSNE.TSNE.Modules import DimensionalityReduction


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

    classifier = DimensionalityReduction(tsne, shuffle, lr=lr)

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
        trainer.fit(classifier, train, [test])
        if model_save:
            tsne.save_model(model_save)
        if test is not None:
            Y = trainer.predict(classifier, test)

    save_results({"o": o, "step": step}, test, Y)


def parse_args():
    available_datasets = []
    if "NeuralTSNE.DatasetLoader.get_datasets" in sys.modules:
        available_datasets = get_datasets._get_available_datasets()
    parser = argparse.ArgumentParser(description="t-SNE Algorithm")
    parser.add_argument(
        "input_file",
        type=FileTypeWithExtensionCheckWithPredefinedDatasets(
            valid_extensions=("txt", "data", "npy"),
            available_datasets=available_datasets,
        ),
        help="Input file",
    )
    parser.add_argument(
        "-iter", type=int, default=1000, help="Number of iterations", required=False
    )
    parser.add_argument(
        "-labels",
        type=FileTypeWithExtensionCheck(valid_extensions=("txt", "data")),
        help="Labels file",
        required=False,
    )
    parser.add_argument(
        "-no_dims", type=int, help="Number of dimensions", required=True, default=2
    )
    parser.add_argument(
        "-perplexity",
        type=float,
        help="Perplexity of the Gaussian kernel",
        required=True,
        default=30.0,
    )
    parser.add_argument(
        "-exclude_cols", type=int, nargs="+", help="Columns to exclude", required=False
    )
    parser.add_argument(
        "-step", type=int, help="Step between samples", required=False, default=1
    )
    parser.add_argument(
        "-exaggeration_iter",
        type=int,
        help="Early exaggeration end",
        required=False,
        default=0,
    )
    parser.add_argument(
        "-exaggeration_value",
        type=float,
        help="Early exaggeration value",
        required=False,
        default=12,
    )
    parser.add_argument(
        "-o", type=str, help="Output filename", required=False, default="result.txt"
    )
    parser.add_argument(
        "-model_save",
        type=str,
        help="Filename to save model to",
        required=False,
    )
    parser.add_argument(
        "-model_load",
        type=str,
        help="Filename to load model from",
        required=False,
    )
    parser.add_argument("-shuffle", action="store_true", help="Shuffle data")
    parser.add_argument(
        "-train_size",
        type=float,
        action=range_action(0, 1),
        help="Train size",
        required=False,
    )
    parser.add_argument(
        "-test_size",
        type=float,
        action=range_action(0, 1),
        help="Test size",
        required=False,
    )
    # parser.add_argument(
    #     "-jobs", type=int, help="Number of jobs", required=False, default=1
    # )
    parser.add_argument(
        "-batch_size", type=int, help="Batch size", required=False, default=1000
    )

    parser.add_argument("-header", action="store_true", help="Data has header")
    parser.add_argument(
        "-net_multipliers",
        type=float,
        nargs="+",
        help="Network multipliers",
        default=[0.75, 0.75, 0.75],
    )
    parser.add_argument("-variance_threshold", type=float, help="Variance threshold")
    parser.add_argument("-cpu", action="store_true", help="Use CPU")
    parser.add_argument(
        "-early_stopping_delta", type=float, help="Early stopping delta", default=1e-5
    )
    parser.add_argument(
        "-early_stopping_patience", type=int, help="Early stopping patience", default=3
    )
    parser.add_argument("-lr", type=float, help="Learning rate", default=1e-3)
    parser.add_argument("-auto_lr", action="store_true", help="Auto learning rate")

    return parser.parse_args()


def main():
    args = parse_args()
    run_tsne(**vars(args))
