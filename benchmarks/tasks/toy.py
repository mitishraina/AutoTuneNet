"""
A tiny synthetic task: 4 Gaussian blobs in 2-D, classified by a small MLP.

It needs no download and trains in milliseconds, so CI and contract tests
can run every method and scenario end to end.
"""
import torch
from torch.utils.data import DataLoader, Dataset, random_split

NUM_CLASSES = 4
CENTRES = torch.tensor([[2.0, 2.0], [-2.0, 2.0], [-2.0, -2.0], [2.0, -2.0]])


class LabelledTensors(Dataset):
    """Like a torchvision dataset: exposes `targets`, so label-noise scenarios work."""

    def __init__(self, data: torch.Tensor, targets: torch.Tensor):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.data[i], int(self.targets[i])


def _blobs(n: int, seed: int) -> LabelledTensors:
    g = torch.Generator().manual_seed(seed)
    targets = torch.randint(0, NUM_CLASSES, (n,), generator=g)
    data = CENTRES[targets] + torch.randn(n, 2, generator=g)
    return LabelledTensors(data, targets)


def get_toy_loaders(batch_size: int = 64, seed: int = 42, val_split: float = 0.2):
    full = _blobs(640, seed=1234)
    val_size = int(len(full) * val_split)
    train_ds, val_ds = random_split(
        full, [len(full) - val_size, val_size], generator=torch.Generator().manual_seed(seed)
    )
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False))


def get_toy_test_loader(batch_size: int = 64):
    return DataLoader(_blobs(256, seed=4321), batch_size=batch_size, shuffle=False)
