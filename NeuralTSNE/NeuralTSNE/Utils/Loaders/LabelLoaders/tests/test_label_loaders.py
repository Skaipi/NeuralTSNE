import io

import pytest
import torch

from NeuralTSNE.Utils.Loaders.LabelLoaders.label_loaders import load_labels


@pytest.mark.parametrize(
    "labels", ["1\n2\n3\n", "3\n2\n1\n6\n", "1\n3\n2\n", "2\n1\n3\n4\n", None]
)
def test_load_labels(labels: str | None):
    if labels is None:
        assert load_labels(labels) is None
    else:
        expected = torch.tensor([float(label) for label in labels.splitlines()])
        labels_file = io.StringIO(labels)
        assert torch.allclose(load_labels(labels_file), expected)
        assert labels_file.closed
