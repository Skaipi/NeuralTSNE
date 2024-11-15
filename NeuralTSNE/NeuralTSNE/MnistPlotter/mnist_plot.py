import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot(
    data: np.ndarray, labels: np.ndarray, is_fashion: bool = False, img_file: str = None
) -> None:
    """
    Plot t-SNE results of mnist dataset.

    Parameters
    ----------
    `data` : `np.ndarray`
        t-SNE data to be plotted.
    `labels` : `np.ndarray`
        Labels corresponding to the data points.
    `is_fashion` : `bool`, optional
        Flag indicating whether the dataset is a fashion dataset.
    `img_file` : `str`, optional
        File path to save the plot as an image.

    Note
    ----
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

    plt.subplots(1, 1)

    sns.scatterplot(
        x=data[:, 0],
        y=data[:, 1],
        hue=map(lambda x: classes[x], labels[: len(data)]),
        palette="Paired",
        legend="full",
    )
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")

    if img_file:
        new_name = img_file
        plt.savefig(new_name)
    plt.show()


def plot_from_file(file: str, labels_file: str, is_fashion: bool = False) -> None:
    """
    Plot t-SNE results of mnist dataset from file.

    Parameters
    ----------
    `file` : `str`
        File path containing t-SNE data.
    `labels_file` : `str`
        File path containing labels data.
    `is_fashion` : `bool`, optional
        Flag indicating whether the dataset is a fashion dataset.

    Note
    ----
    This function reads t-SNE data and labels from files and plots the results using the `plot` function.
    """
    data = None

    with open(file, "r") as f:
        _ = int(f.readline())
        data = np.loadtxt(f)

    labels = None
    if labels_file:
        with open(labels_file, "r") as f:
            labels = np.loadtxt(f, dtype="int")

    plot(data, labels, is_fashion, file.rsplit(".", 1)[0] + ".png")
