from unittest.mock import MagicMock, call, patch, mock_open

import pytest
import numpy as np
import io

import NeuralTSNE.MnistPlotter as plotter


@pytest.mark.parametrize("is_fashion", [True, False])
@patch("matplotlib.pyplot.subplots")
@patch("matplotlib.pyplot.xlabel")
@patch("matplotlib.pyplot.ylabel")
@patch("seaborn.scatterplot")
@patch("matplotlib.pyplot.savefig")
@patch("matplotlib.pyplot.show")
def test_plot(
    mock_plt_show: MagicMock,
    mock_plt_savefig: MagicMock,
    mock_scatterplot: MagicMock,
    mock_plt_ylabel: MagicMock,
    mock_plt_xlabel: MagicMock,
    mock_plt_subplots: MagicMock,
    is_fashion: bool,
):
    img_file = "test.png"
    mock_plt_subplots.return_value = (None, None)
    plotter.plot(np.array([[1, 2], [3, 4]]), np.array([1, 2]), is_fashion, img_file)

    mock_scatterplot.assert_called_once()
    mock_plt_savefig.assert_called_once_with(img_file)
    mock_plt_show.assert_called_once()
    mock_plt_xlabel.assert_called_once()
    mock_plt_ylabel.assert_called_once()
    mock_plt_subplots.assert_called_once()


@pytest.mark.parametrize("input_file", ["test_data.txt"])
@pytest.mark.parametrize("labels_file", ["test_labels.txt", None])
@pytest.mark.parametrize("is_fashion", [True, False])
@patch("builtins.open", new_callable=mock_open)
@patch("numpy.loadtxt")
@patch("NeuralTSNE.MnistPlotter.mnist_plot.plot")
def test_plot_from_file(
    mock_plot: MagicMock,
    mock_loadtxt: MagicMock,
    mock_files: MagicMock,
    input_file: str,
    labels_file: str | None,
    is_fashion: bool,
):
    file_content = "10\n1 2 3\n4 5 6"
    labels_content = "1\n2"

    handlers = (
        [
            io.StringIO(file_content),
            io.StringIO(labels_content),
        ]
        if labels_file
        else [io.StringIO(file_content)]
    )

    mock_files.side_effect = handlers

    data_list = list(line.split() for line in file_content.splitlines()[1:])
    data = np.array(data_list, dtype="float32")

    labels_list = list(line.split() for line in labels_content.splitlines())
    labels = np.array(labels_list, dtype="int32") if labels_file else None

    mock_loadtxt.side_effect = [data, labels]

    plotter.plot_from_file(input_file, labels_file, is_fashion)

    if labels_file:
        mock_loadtxt.assert_has_calls(
            [call(handlers[0]), call(handlers[1])],
            any_order=True,
        )
        mock_files.assert_has_calls(
            [call(input_file, "r"), call(labels_file, "r")], any_order=True
        )
    else:
        mock_loadtxt.assert_called_once_with(handlers[0])
        mock_files.assert_called_once_with(input_file, "r")

    mock_plot.assert_called_once_with(
        data, labels, is_fashion, input_file.rsplit(".", 1)[0] + ".png"
    )
