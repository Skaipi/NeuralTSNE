from typing import List, Optional
from collections import OrderedDict

from torch import nn


class NeuralNetwork(nn.Module):
    """
    Neural network model for dimensionality reduction.

    Parameters
    ----------
    `initial_features` : `int`, optional
        Number of input features.
    `n_components` : `int`, optional
        Number of components in the output.
    `multipliers` : `List[float]`, optional
        List of multipliers for hidden layers.
    `pre_filled_layers` : `OderedDict`, optional
        Pre-filled OrderedDict for layers. Defaults to `None`.

    Note
    ----
    The neural network is designed for dimensionality reduction with
    hidden layers defined by the list of multipliers. ReLU activation
    functions are applied between layers.
    If `pre_filled_layers` is provided, the neural network is initialized with
    the given layers and other parameters are ignored.
    """

    def __init__(
        self,
        initial_features: int | None = None,
        n_components: int | None = None,
        multipliers: List[float] | None = None,
        pre_filled_layers: Optional[OrderedDict] = None,
    ) -> None:
        super(NeuralNetwork, self).__init__()

        if pre_filled_layers is not None:
            self.linear_relu_stack = nn.Sequential(pre_filled_layers)
            return

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

        Parameters
        ----------
        `x` : `torch.Tensor`
            Input tensor.

        Returns
        -------
        `torch.Tensor`
            Output tensor.
        """
        logits = self.linear_relu_stack(x)
        return logits
