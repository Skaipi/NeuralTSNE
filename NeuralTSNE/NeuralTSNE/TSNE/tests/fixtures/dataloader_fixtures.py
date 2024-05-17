from unittest.mock import patch

import pytest
from torch.utils.data import DataLoader, Dataset


@pytest.fixture
def mock_dataloaders():
    with patch(
        "NeuralTSNE.TSNE.ParametricTSNE.parametric_tsne.ParametricTSNE.create_dataloaders",
        autospec=True,
    ) as mock_create_dataloaders:
        mock_create_dataloaders.return_value = DataLoader(Dataset()), DataLoader(
            Dataset()
        )
        yield mock_create_dataloaders
