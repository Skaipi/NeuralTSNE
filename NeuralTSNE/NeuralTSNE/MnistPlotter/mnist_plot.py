import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from NeuralTSNE.TSNE import FileTypeWithExtensionCheck


def plot(data, labels, is_fashion=False, img_file=None):
    """
    Plot t-SNE results of mnist dataset.

    ---
    ### Parameters:
        - `data`: t-SNE data to be plotted.
        - `labels`: Labels corresponding to the data points.
        - `is_fashion` (`bool`, optional): Flag indicating whether the dataset is a fashion dataset.
        - `img_file` (`str`, optional): File path to save the plot as an image.

    This function plots the t-SNE results with colored points based on the provided labels.
    """
    if is_fashion:
        classes = [
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ]
    else:
        classes = [i for i in range(10)]

    fig, ax = plt.subplots(1, 1)

    sns.scatterplot(
        x=data[:, 0],
        y=data[:, 1],
        hue=labels[: len(data)],
        palette="Paired",
        legend="full",
    )
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")

    if img_file:
        new_name = img_file
        plt.savefig(new_name)
    plt.show()


def plot_from_file(file, labels_file, is_fashion=False):
    """
    Plot t-SNE results of mnist dataset from file.

    ---
    ### Parameters:
        - `file` (`str`): File path containing t-SNE data.
        - `labels_file` (`str`): File path containing labels data.
        - `is_fashion` (`bool`, optional): Flag indicating whether the dataset is a fashion dataset.

    This function reads t-SNE data and labels from files and plots the results using the `plot` function.
    """
    data = None

    with open(file, "r") as f:
        _ = int(f.readline())
        data = np.loadtxt(f)

    labels = None
    if labels_file:
        with open(labels_file, "r") as f:
            labels = np.loadtxt(f)

    plot(data, labels, is_fashion, file.rsplit(".", 1)[0] + ".png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plotter for t-SNE results from MNIST datasets"
    )
    parser.add_argument("input_file", type=str, help="Input file")
    parser.add_argument(
        "-labels",
        type=FileTypeWithExtensionCheck(valid_extensions=("txt", "data")),
        help="Labels file",
        required=False,
    )
    parser.add_argument(
        "-col", type=int, help="Column from label file", required=False, default=None
    )
    parser.add_argument("-step", type=int, help="Step", required=False, default=1)
    parser.add_argument(
        "-alpha", type=float, help="Alpha of point", required=False, default=1
    )
    parser.add_argument("-f", action="store_true", help="Fashion MNIST")

    args = parser.parse_args()

    data = None
    step = None

    with open(args.input_file, "r") as f:
        step = int(f.readline())
        data = np.loadtxt(f)

    labels = None
    if args.labels:
        labels = np.loadtxt(args.labels)
        args.labels.close()

    if args.f:
        classes = [
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ]
    else:
        classes = [i for i in range(10)]

    fig, ax = plt.subplots(1, 1)

    sns.scatterplot(
        x=data[:, 0], y=data[:, 1], hue=labels[: len(data)], palette="Paired"
    )
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(classes)

    new_name = args.input_file.rsplit(".", 1)[0] + ".png"

    plt.savefig(new_name)
    plt.show()
