import itertools
from unittest.mock import MagicMock, call, patch, mock_open

import pytest
from parameterized import parameterized

import NeuralTSNE.Plotter as plotter


@parameterized.expand([True, False])
@patch("NeuralTSNE.Plotter.plot.sns")
@patch("NeuralTSNE.Plotter.plot.plt")
def test_plot(is_fashion: bool, mock_plt: MagicMock, mock_sns: MagicMock):
    pass


@patch("builtins.open", new_callable=mock_open)
def test_plot_from_file(mock_open: MagicMock):
    pass
