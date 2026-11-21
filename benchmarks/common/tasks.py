from benchmarks.common.dataset import (
    get_mnist_loaders,
    get_cifar_loaders,
    get_mnist_test_loader,
    get_cifar_test_loader,
)
from benchmarks.common.model import MNISTCNN, CIFARCNN

# task name -> (train/val loader factory, test loader factory, model class)
TASKS = {
    "mnist": (get_mnist_loaders, get_mnist_test_loader, MNISTCNN),
    "cifar10": (get_cifar_loaders, get_cifar_test_loader, CIFARCNN),
}

# Per-task training defaults shared by the legacy benchmark scripts. They match
# the AutoTuneNet-Bench configs for the same task.
# lr_range is the search range shared by AutoTuneNet and offline HPO.
TASK_DEFAULTS = {
    "mnist": {"batch_size": 64, "lr": 0.01, "lr_range": (1e-4, 1e-1)},
    "cifar10": {"batch_size": 128, "lr": 0.001, "lr_range": (1e-4, 1e-3)},
}

# Accepted spellings found in configs
ALIASES = {
    "mnist_cnn": "mnist",
    "cifar": "cifar10",
    "cifar_cnn": "cifar10",
    "cifar10_cnn": "cifar10",
}


def resolve_task(name: str) -> str:
    key = name.strip().lower()
    key = ALIASES.get(key, key)
    if key not in TASKS:
        raise ValueError(f"Unknown task '{name}'. Available: {sorted(TASKS)}")
    return key


def build_task(name: str, batch_size: int, seed: int):
    """Return (train_loader, val_loader, model) for a benchmark task."""
    loaders_fn, _, model_cls = TASKS[resolve_task(name)]
    train_loader, val_loader = loaders_fn(batch_size=batch_size, seed=seed)
    return train_loader, val_loader, model_cls()


def build_test_loader(name: str, batch_size: int):
    """Held-out test split for a task (reporting only, never tuning)."""
    _, test_fn, _ = TASKS[resolve_task(name)]
    return test_fn(batch_size=batch_size)


def build_model(name: str):
    return TASKS[resolve_task(name)][2]()
