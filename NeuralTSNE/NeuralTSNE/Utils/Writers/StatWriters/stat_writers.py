from typing import Any, List, Union

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def save_means_and_vars(data: torch.Tensor, filtered_data: torch.Tensor = None) -> None:
    """
    Calculate and save the means and variances of columns in a 2D `torch.Tensor` to a file.

    If `filtered_data` is provided, it calculates and saves means and variances for both original and filtered columns.

    ---
    ### Parameters:
        - `data` (`torch.Tensor`): The input 2D tensor for which means and variances are calculated.
        - `filtered_data` (`torch.Tensor`, optional): A filtered version of the input data. Defaults to `None`.

    ---
    ### Note:
    - The function calculates means and variances for each column in the input data.
    - If `filtered_data` is provided, it also calculates and saves means and variances for the corresponding filtered columns.
    """
    means = data.mean(axis=0)
    variances = data.var(axis=0)

    if filtered_data is not None:
        filtered_means = filtered_data.mean(axis=0)
        filtered_variances = filtered_data.var(axis=0)

    with open("means_and_vars.txt", "w") as f:
        f.writelines("column\tmean\tvar\n")
        for v in range(len(means)):
            f.writelines(f"{v}\t{means[v]}\t{variances[v]}\n")
        if filtered_data is not None:
            f.writelines("\nfiltered_column\tfiltered_mean\tfiltered_var\n")
            for v in range(len(filtered_means)):
                f.writelines(f"{v}\t{filtered_means[v]}\t{filtered_variances[v]}\n")


def save_results(args: dict, test: DataLoader, Y: Union[List[Any], List[List[Any]]]):
    """
    Save results to a file.

    ---
    ### Parameters:
        - `args` (`dict`): Dictionary containing arguments, including the output file path (`o`) and step size (`step`).
        - `test` (`DataLoader`): DataLoader for the test dataset.
        - `Y` (`Union[List[Any], List[List[Any]]]`): List of results to be saved.

    This function saves the results to a file specified by the output file path in the arguments.
    """
    if test is not None:
        with open(args["o"], "w") as f:
            f.writelines(f"{args['step']}\n")
            for _, batch in tqdm(
                enumerate(Y), unit="batches", total=(len(Y)), desc="Saving results"
            ):
                for entry in batch:
                    output_line = "\t".join([str(x) for x in entry])
                    f.writelines(f"{output_line}\n")
