"""Built-in benchmark tasks. Register a new one with TASKS.add(name, TaskSpec(...))."""
from autotunenet.bench.core.registry import TASKS
from autotunenet.bench.tasks.base import TaskSpec
from torchvision import datasets

from autotunenet.bench.tasks.data import (
    load_dataset,
    get_cifar_loaders,
    get_cifar_test_loader,
    get_mnist_loaders,
    get_mnist_test_loader,
)
from autotunenet.bench.tasks.models import CIFARCNN, MNISTCNN, ToyMLP
from autotunenet.bench.tasks.toy import get_toy_loaders, get_toy_test_loader

def _fetch(dataset_cls):
    def fetch():
        for train in (True, False):
            load_dataset(dataset_cls, train=train)
    return fetch


TASKS.add("mnist", TaskSpec(
    name="mnist",
    build_loaders=lambda batch_size, seed: get_mnist_loaders(batch_size=batch_size, seed=seed),
    build_test_loader=lambda batch_size: get_mnist_test_loader(batch_size=batch_size),
    build_model=MNISTCNN,
    batch_size=64,
    lr=0.01,
    lr_range=(1e-4, 1e-1),
    data_paths=("MNIST/raw",),
    prepare_data=_fetch(datasets.MNIST),
))

TASKS.add("cifar10", TaskSpec(
    name="cifar10",
    build_loaders=lambda batch_size, seed: get_cifar_loaders(batch_size=batch_size, seed=seed),
    build_test_loader=lambda batch_size: get_cifar_test_loader(batch_size=batch_size),
    build_model=CIFARCNN,
    batch_size=128,
    lr=0.001,
    lr_range=(1e-4, 1e-3),
    data_paths=("cifar-10-batches-py",),
    prepare_data=_fetch(datasets.CIFAR10),
))

TASKS.add("toy", TaskSpec(
    name="toy",
    build_loaders=lambda batch_size, seed: get_toy_loaders(batch_size=batch_size, seed=seed),
    build_test_loader=lambda batch_size: get_toy_test_loader(batch_size=batch_size),
    build_model=ToyMLP,
    batch_size=64,
    lr=0.01,
    lr_range=(1e-4, 1e-1),
))
