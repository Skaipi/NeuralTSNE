import importlib as _importlib
from . import TSNE
from . import DatasetLoader
from . import Plotter
from . import MnistPlotter
from . import Utils

__version__ = "{{VERSION}}"

submodules = ["TSNE", "DatasetLoader", "Plotter", "MnistPlotter", "Utils"]

__all__ = submodules + ["__version__"]


def __dir__():
    return __all__


def __getattr__(name):
    if name in submodules:
        return _importlib.import_module(f"NeuralTSNE.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
