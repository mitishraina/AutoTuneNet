from benchmarks.common.dataset import get_mnist_loaders, get_cifar_loaders
from benchmarks.common.model import MNISTCNN, CIFARCNN

# task name -> (data loader factory, model class)
TASKS = {
    "mnist": (get_mnist_loaders, MNISTCNN),
    "cifar10": (get_cifar_loaders, CIFARCNN),
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
    loaders_fn, model_cls = TASKS[resolve_task(name)]
    train_loader, val_loader = loaders_fn(batch_size=batch_size, seed=seed)
    return train_loader, val_loader, model_cls()
