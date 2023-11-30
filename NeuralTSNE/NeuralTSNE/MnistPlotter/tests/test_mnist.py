import itertools
from unittest.mock import MagicMock, call, patch, mock_open

import pytest
from parameterized import parameterized

import NeuralTSNE.MnistPlotter as plotter


@parameterized.expand([True, False])
@patch("NeuralTSNE.MnistPlotter.mnist_plot.sns")
@patch("NeuralTSNE.MnistPlotter.mnist_plot.plt")
def test_plot(is_fashion: bool, mock_plt: MagicMock, mock_sns: MagicMock):
    pass


@patch("builtins.open", new_callable=mock_open)
def test_plot_from_file(mock_open: MagicMock):
    pass
