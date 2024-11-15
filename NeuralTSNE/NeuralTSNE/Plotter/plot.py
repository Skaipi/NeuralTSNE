from typing import List
import matplotlib.pyplot as plt
import numpy as np


def plot(
    data: np.ndarray,
    labels: np.ndarray | None,
    step: int,
    marker_size: int,
    alpha: float,
    are_neural_labels: bool = False,
    img_file: str | None = None,
    kwargs: dict | None = None,
) -> None:
    """
    Plot t-SNE results.

    Parameters
    ----------
    `data` : `np.ndarray`
        t-SNE data to be plotted.
    `labels` : `np.ndarray`, optional
        Labels corresponding to the data points.
    `step` : `int`
        Step size for subsampling the data.
    `marker_size` : `int`
        Marker size for the scatter plot.
    `alpha` : `float`
        Alpha value for transparency in the scatter plot.
    `are_neural_labels` : `bool`, optional
        Flag indicating whether the labels are neural network predictions.
    `img_file` : `str`, optional
        File path to save the plot as an image.
    `**kwargs` : `dict`, optional
        Additional keyword arguments.

    Important
    ---------
    The following additional keyword arguments are available:

    `file_step` : `int`, optional
        Step size for subsampling labels. Defaults to `1`.

    Note
    ----
    This function plots the t-SNE results with scatter plot, allowing customization of various plot parameters.
    """
    if kwargs is None:
        kwargs = {}
    f_step = kwargs.get("file_step", 1)

    plt.subplots(1, 1)

    (
        plt.scatter(
            data[::step, 0],
            data[::step, 1],
            marker_size,
            alpha=alpha,
            marker=".",
        )
        if labels is None
        else plt.scatter(
            data[::step, 0],
            data[::step, 1],
            marker_size,
            labels[:: f_step * step] if not are_neural_labels else labels[::step],
            alpha=alpha,
            marker=".",
        )
    )

    plt.ylabel("t-SNE 2")
    plt.xlabel("t-SNE 1")

    if img_file:
        new_name = img_file
        plt.savefig(new_name)
    plt.show()


def plot_from_file(
    file: str,
    labels_file: str,
    columns: List[int],
    step: int,
    marker_size: int,
    alpha: float,
    are_neural_labels: bool = False,
) -> None:
    """
    Plot t-SNE results from file.

    Parameters
    ----------
    `file` : `str`
        File path containing t-SNE data.
    `labels_file` : `str`
        File path containing labels data.
    `columns` : `List[int]`
        Column indices to load from the labels file.
    `step` : `int`
        Step size for subsampling the data.
    `marker_size` : `int`
        Marker size for the scatter plot.
    `alpha` : `float`
        Alpha value for transparency in the scatter plot.
    `are_neural_labels` : `bool`, optional
        Flag indicating whether the labels are neural network predictions.

    Note
    ----
    This function reads t-SNE data and labels from files, applies subsampling, and plots the results using the `plot` function.
    """
    data = None
    file_step = None

    with open(file, "r") as f:
        file_step = int(f.readline())
        data = np.loadtxt(f)

    labels = None
    if labels_file:
        with open(labels_file, "r") as f:
            labels = np.loadtxt(f, usecols=columns, dtype="int")
        data = data[: len(labels)]

    plot(
        data,
        labels,
        step,
        marker_size,
        alpha,
        are_neural_labels,
        file.rsplit(".", 1)[0] + ".png",
        {"file_step": file_step},
    )
