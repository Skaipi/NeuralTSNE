import io
from typing import Tuple

import torch
from torch.utils.data import Dataset


class PersistentStringIO(io.StringIO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._closed = False

    def close(self):
        self._closed = True

    @property
    def closed(self):
        return self._closed


class MyDataset(Dataset):
    def __init__(
        self,
        num_samples: int,
        num_variables: int,
        item_range: Tuple[float, float] | Tuple[int, int] = None,
        generate_int: bool = False,
    ):
        self.num_samples = num_samples
        self.num_variables = num_variables
        self.item_range = item_range or (0, 1)
        self.generate_int = generate_int

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        if self.generate_int:
            sample = torch.randint(*self.item_range, size=(self.num_variables,))
        else:
            sample = torch.FloatTensor(self.num_variables).uniform_(*self.item_range)
        return tuple([sample])


class DataLoaderMock:
    def __init__(self, dataset: MyDataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size
        self.batches = []

    def __iter__(self):
        for i in range(0, len(self.dataset), self.batch_size):
            batch = tuple(
                torch.cat(
                    [
                        torch.unsqueeze(self.dataset[j][k], 0)
                        for j in range(i, i + self.batch_size)
                    ],
                    dim=0,
                )
                for k in range(len(self.dataset[0]))
            )
            self.batches.append(batch)
            yield batch

    def __len__(self):
        return len(self.dataset)
