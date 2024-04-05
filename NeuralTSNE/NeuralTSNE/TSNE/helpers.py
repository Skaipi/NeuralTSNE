from typing import Tuple
import torch


def Hbeta(D: torch.Tensor, beta: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate entropy and probability distribution based on a distance matrix.

    ---
    ### Parameters:
        - `D` (`torch.Tensor`): Distance matrix.
        - `beta` (`float`): Parameter for the computation.

    ---
    ### Returns:
        - `Tuple[torch.Tensor, torch.Tensor]`: Entropy and probability distribution.

    The function calculates the entropy and probability distribution based on
    the provided distance matrix (`D`) and the specified parameter (`beta`).
    """
    P = torch.exp(-D * beta)
    sumP = torch.sum(P)
    H = torch.log(sumP) + beta * torch.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p_job(
    data: Tuple[int, torch.Tensor, torch.Tensor],
    tolerance: float,
    max_iterations: int = 50,
) -> Tuple[int, torch.Tensor, torch.Tensor, int]:
    """
    Perform a binary search to find an appropriate value of `beta` for a given point.

    ---
    ### Parameters:
        - `data` (`Tuple[int, torch.Tensor, torch.Tensor]`): Tuple containing index, distance matrix, and target entropy.
        - `tolerance` (`float`): Tolerance level for convergence.
        - `max_iterations` (`int`, optional): Maximum number of iterations for the binary search. Default is 50.

    ---
    ### Returns:
        - `Tuple[int, torch.Tensor]`: Index, probability distribution, entropy difference, and number of iterations.

    The function performs a binary search to find an appropriate value of `beta` for a given point,
    aiming to match the target entropy. It returns the index, probability distribution, entropy difference, and number of iterations.
    """
    i, Di, logU = data
    beta = 1.0
    beta_min = -torch.inf
    beta_max = torch.inf

    H, thisP = Hbeta(Di, beta)
    Hdiff = H - logU

    it = 0
    while it < max_iterations and torch.abs(Hdiff) > tolerance:
        if Hdiff > 0:
            beta_min = beta
            if torch.isinf(torch.tensor(beta_max)):
                beta *= 2
            else:
                beta = (beta + beta_max) / 2
        else:
            beta_max = beta
            if torch.isinf(torch.tensor(beta_min)):
                beta /= 2
            else:
                beta = (beta + beta_min) / 2

        H, thisP = Hbeta(Di, beta)
        Hdiff = H - logU
        it += 1
    return i, thisP, Hdiff, it


def x2p(
    X: torch.Tensor,
    perplexity: int,
    tolerance: float,
) -> torch.Tensor:
    """
    Compute conditional probabilities using the t-SNE algorithm.

    ---
    ### Parameters:
        - `X` (`torch.Tensor`): Input data tensor.
        - `perplexity` (`int`): Perplexity parameter for t-SNE.
        - `tolerance` (`float`): Tolerance level for convergence.

    ---
    ### Returns:
        - `torch.Tensor`: Conditional probability matrix.
    """
    n = X.shape[0]
    logU = torch.log(torch.tensor([perplexity], device=X.device))

    sum_X = torch.sum(torch.square(X), dim=1)
    D = torch.add(torch.add(-2 * torch.mm(X, X.mT), sum_X).T, sum_X)

    idx = (1 - torch.eye(n)).type(torch.bool)
    D = D[idx].reshape((n, -1))

    P = torch.zeros(n, n, device=X.device)

    for i in range(n):
        P[i, idx[i]] = x2p_job((i, D[i], logU), tolerance)[1]
    return P
