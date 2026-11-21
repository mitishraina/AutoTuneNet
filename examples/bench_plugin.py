"""
A benchmark plugin: one task (Fashion-MNIST with a small CNN) and one scenario
(input noise from a given epoch). Register them, then pass this file with --plugins:

    autotunenet list --plugins examples/bench_plugin.py
    autotunenet run --experiment examples/bench_experiment.yaml --plugins examples/bench_plugin.py
"""
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from autotunenet.bench.core.registry import SCENARIOS, TASKS
from autotunenet.bench.scenarios.base import Scenario
from autotunenet.bench.tasks.base import TaskSpec
from autotunenet.bench.tasks.data import load_dataset

# --8<-- [start:task]
_TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])


def _loaders(batch_size: int, seed: int):
    full = load_dataset(datasets.FashionMNIST, train=True, transform=_TRANSFORM)
    val_size = len(full) // 10
    train, val = random_split(full, [len(full) - val_size, val_size],
                              generator=torch.Generator().manual_seed(seed))   # same split for every method
    return DataLoader(train, batch_size, shuffle=True), DataLoader(val, batch_size)


def _test_loader(batch_size: int):
    return DataLoader(load_dataset(datasets.FashionMNIST, train=False, transform=_TRANSFORM), batch_size)


def _fetch():
    for train in (True, False):
        load_dataset(datasets.FashionMNIST, train=train)


def _model():
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10),
    )


TASKS.add("fashion_mnist", TaskSpec(
    name="fashion_mnist",
    build_loaders=_loaders,
    build_test_loader=_test_loader,
    build_model=_model,
    batch_size=64,
    lr=1e-3,                          # initial learning rate (Adam)
    lr_range=(1e-4, 1e-2),            # every tuning method searches this, log scale
    data_paths=("FashionMNIST/raw",), # what to upload for hosted runtimes, relative to the data dir
    prepare_data=_fetch,
))
# --8<-- [end:task]


# --8<-- [start:scenario]
@SCENARIOS.register("input_noise")
class InputNoise(Scenario):
    """From `start_epoch`, Gaussian noise with standard deviation `std` is added to every input."""

    def __init__(self, start_epoch: int = 5, std: float = 0.5, seed: int = 0):
        self.shift_epoch = start_epoch
        self.std = std
        self.generator = torch.Generator().manual_seed(seed)

    def _add_noise(self, module, inputs):
        x = inputs[0]
        noise = torch.randn(x.shape, generator=self.generator).to(x.device)
        return (x + self.std * noise,)

    def on_epoch_start(self, epoch: int, trainer) -> None:
        if epoch == self.shift_epoch:
            trainer.model.register_forward_pre_hook(self._add_noise)
# --8<-- [end:scenario]
