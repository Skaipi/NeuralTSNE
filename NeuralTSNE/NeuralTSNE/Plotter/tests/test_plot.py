import io
from unittest.mock import MagicMock, call, mock_open, patch

import numpy as np
import pytest

import NeuralTSNE.Plotter as plotter


@pytest.mark.parametrize("are_neural_labels", [True, False])
@pytest.mark.parametrize("file_step", [None, 2])
@pytest.mark.parametrize("step", [1, 3])
@patch("matplotlib.pyplot.subplots")
@patch("matplotlib.pyplot.xlabel")
@patch("matplotlib.pyplot.ylabel")
@patch("matplotlib.pyplot.scatter")
@patch("matplotlib.pyplot.savefig")
@patch("matplotlib.pyplot.show")
def test_plot(
    mock_plt_show: MagicMock,
    mock_plt_savefig: MagicMock,
    mock_plt_scatter: MagicMock,
    mock_plt_ylabel: MagicMock,
    mock_plt_xlabel: MagicMock,
    mock_plt_subplots: MagicMock,
    step: int,
    file_step: int | None,
    are_neural_labels: bool,
):
    img_file = "test.png"
    mock_plt_subplots.return_value = (None, None)
    plotter.plot(
        np.array([[1, 2], [3, 4], [5, 6], [7, 8]]),
        np.array([1, 2, 3, 4]),
        step,
        2,
        1,
        are_neural_labels,
        img_file,
        {"file_step": file_step} if file_step else None,
    )

    mock_plt_scatter.assert_called_once()
    mock_plt_savefig.assert_called_once_with(img_file)
    mock_plt_show.assert_called_once()
    mock_plt_xlabel.assert_called_once()
    mock_plt_ylabel.assert_called_once()
    mock_plt_subplots.assert_called_once()


@pytest.mark.parametrize("input_file", ["test_data.txt"])
@pytest.mark.parametrize("labels_file", ["test_labels.txt", None])
@pytest.mark.parametrize("step", [1, 3])
@pytest.mark.parametrize("marker_size", [1, 3])
@pytest.mark.parametrize("alpha", [1, 0.5])
@pytest.mark.parametrize("columns", [None, 2])
@pytest.mark.parametrize("are_neural_labels", [True, False])
@patch("builtins.open", new_callable=mock_open)
@patch("numpy.loadtxt")
@patch("NeuralTSNE.Plotter.plot.plot")
def test_plot_from_file(
    mock_plot: MagicMock,
    mock_loadtxt: MagicMock,
    mock_files: MagicMock,
    are_neural_labels: bool,
    columns: int | None,
    alpha: float,
    marker_size: float,
    step: int,
    labels_file: str | None,
    input_file: str,
):
    file_content = "20\n1 2 3\n4 5 6\n7 8 9\n10 11 12"
    labels_content = "1 4\n2 5\n3 6"

    handlers = (
        [
            io.StringIO(file_content),
            io.StringIO(labels_content),
        ]
        if labels_file
        else [io.StringIO(file_content)]
    )

    mock_files.side_effect = handlers

    data_list = list(line.split() for line in file_content.splitlines())
    data = np.array(data_list[1:], dtype="float32")

    labels_list = list(line.split() for line in labels_content.splitlines())
    labels = np.array(labels_list, dtype="int32") if labels_file else None

    mock_loadtxt.side_effect = [data, labels]

    file_step = int(data_list[0][0])
    if labels_file:
        data = data[: len(labels)]

    plotter.plot_from_file(
        input_file,
        labels_file,
        columns,
        step,
        marker_size,
        alpha,
        are_neural_labels,
    )

    if labels_file:
        mock_loadtxt.assert_has_calls(
            [
                call(handlers[0]),
                call(handlers[1], usecols=columns, dtype="int"),
            ],
            any_order=True,
        )
        mock_files.assert_has_calls(
            [call(input_file, "r"), call(labels_file, "r")], any_order=True
        )
    else:
        mock_loadtxt.assert_called_once_with(handlers[0])
        mock_files.assert_called_once_with(input_file, "r")

    mock_plot.assert_called_once()

    mock_plot_args = mock_plot.call_args[0]
    np.testing.assert_array_equal(mock_plot_args[0], data)
    np.testing.assert_array_equal(mock_plot_args[1], labels)
    assert mock_plot_args[2] == step
    assert mock_plot_args[3] == marker_size
    assert mock_plot_args[4] == alpha
    assert mock_plot_args[5] == are_neural_labels
    assert mock_plot_args[6] == input_file.rsplit(".", 1)[0] + ".png"
    assert mock_plot_args[7] == {"file_step": file_step}
