"""Built-in benchmark tasks. Register a new one with TASKS.add(name, TaskSpec(...))."""
from benchmarks.core.registry import TASKS
from benchmarks.tasks.base import TaskSpec
from benchmarks.tasks.data import (
    get_cifar_loaders,
    get_cifar_test_loader,
    get_mnist_loaders,
    get_mnist_test_loader,
)
from benchmarks.tasks.models import CIFARCNN, MNISTCNN, ToyMLP
from benchmarks.tasks.toy import get_toy_loaders, get_toy_test_loader

TASKS.add("mnist", TaskSpec(
    name="mnist",
    build_loaders=lambda batch_size, seed: get_mnist_loaders(batch_size=batch_size, seed=seed),
    build_test_loader=lambda batch_size: get_mnist_test_loader(batch_size=batch_size),
    build_model=MNISTCNN,
    batch_size=64,
    lr=0.01,
    lr_range=(1e-4, 1e-1),
))

TASKS.add("cifar10", TaskSpec(
    name="cifar10",
    build_loaders=lambda batch_size, seed: get_cifar_loaders(batch_size=batch_size, seed=seed),
    build_test_loader=lambda batch_size: get_cifar_test_loader(batch_size=batch_size),
    build_model=CIFARCNN,
    batch_size=128,
    lr=0.001,
    lr_range=(1e-4, 1e-3),
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
