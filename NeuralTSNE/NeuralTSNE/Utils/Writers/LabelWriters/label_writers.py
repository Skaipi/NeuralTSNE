from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def save_torch_labels(output: str, test: Dataset) -> None:
    """
    Save labels from a `torch.Dataset` to a text file.

    The function extracts labels from the provided `test` dataset and saves them to a text file.
    The output file is named based on the provided `output` parameter.

    Parameters
    ----------
    `output` : `str`
        The output file path for saving labels.
    `test` : `Dataset`
        The `torch.Dataset` containing labels to be saved.

    Note
    ----
    - The function iterates through the `test` dataset, extracts labels, and saves them to a text file.
    - The output file is named by appending `"_labels.txt"` to the `output` parameter, removing the file extension if present.
    """
    with open(
        output.rsplit(".", maxsplit=1)[0] + "_labels.txt",
        "w",
    ) as f:
        for _, row in tqdm(
            enumerate(test), unit="samples", total=len(test), desc="Saving labels"
        ):
            f.writelines(f"{row[1]}\n")


def save_labels_data(
    args: dict,
    test: DataLoader,
) -> None:
    """
    Save labels data to a new file.

    Parameters
    ----------
    `args` : `dict`
        Dictionary containing arguments, including the output file path (`o`).
    `test` : `DataLoader`
        DataLoader for the test dataset.

    Note
    ----
    This function saves the labels data to a new file with a name based on the original output file path.
    """
    if test is not None:
        new_name = args["o"].rsplit(".", 1)[0] + "_labels.txt"
        with open(new_name, "w") as f:
            for _, batch in tqdm(
                enumerate(test),
                unit="batches",
                total=(len(test)),
                desc="Saving new labels",
            ):
                for samples in batch:
                    samples = samples.tolist()
                    for sample in samples:
                        for col in sample:
                            f.write(str(col))
                            f.write("\t")
                        f.write("\n")
