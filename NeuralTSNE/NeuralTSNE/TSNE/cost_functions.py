from typing import Any
import torch


class CostFunctions:
    @staticmethod
    def __call__(name):
        """
        Returns the specified cost function by name.

        ---
        ### Parameters:
            - `name` (`str`): The name of the cost function to retrieve.

        ---
        ### Returns:
            - `callable`: The specified cost function.

        """
        return getattr(CostFunctions, name)

    @staticmethod
    def kl_divergence(
        Y: torch.Tensor, P: torch.Tensor, params: dict[str, Any]
    ) -> torch.Tensor:
        """
        Calculate the Kullback-Leibler divergence.

        ---
        ### Parameters:
            - `Y` (`torch.Tensor`): Embedding tensor.
            - `P` (`torch.Tensor`): Conditional probability matrix.

        ---
        ### Returns:
            - `torch.Tensor`: Kullback-Leibler divergence.

        Calculates the Kullback-Leibler divergence between the true conditional probability matrix P
        and the conditional probability matrix Q based on the current embedding Y.
        """
        sum_Y = torch.sum(torch.square(Y), dim=1)
        eps = torch.tensor([1e-15], device=params["device"])
        D = (
            sum_Y + torch.reshape(sum_Y, [-1, 1]) - 2 * torch.matmul(Y, Y.mT)
        )  # TODO: cdist may be the way to calculate distances neatly
        Q = torch.pow(1 + D / 1.0, -(1.0 + 1) / 2)
        Q *= 1 - torch.eye(params["batch_size"], device=params["device"])
        Q /= torch.sum(Q)
        Q = torch.maximum(Q, eps)
        C = torch.log((P + eps) / (Q + eps))
        C = torch.sum(P * C)
        return C
