"""TaskSpec: what a benchmark task (a dataset and a model) must provide."""
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class TaskSpec:
    """
    A dataset and model pair, with its training defaults.

    Register one with `TASKS.add(name, TaskSpec(...))`, in a plugin file or
    in `autotunenet.bench.tasks`.

    Attributes:
        name: Task name, as used in experiment files.
        build_loaders: `(batch_size, seed) -> (train_loader, val_loader)`. The
            train/validation split must depend only on `seed`, so every method
            sees the same split.
        build_test_loader: `(batch_size) -> test_loader`, the held-out split,
            used once to report the final model.
        build_model: `() -> nn.Module`, a freshly initialised model.
        batch_size: Batch size used by every method.
        lr: Default initial learning rate.
        lr_range: `(low, high)` learning-rate range searched (log scale) by
            every tuning method.
        data_paths: Files or folders the task reads, relative to the data
            directory (`autotunenet.bench.paths.data_dir()`). Hosted runtimes
            upload these, e.g. `("MNIST/raw",)`.
        prepare_data: `() -> None`, downloads the data into the data directory
            if missing; used by `autotunenet data`.

    Examples:
        >>> TASKS.add("my_task", TaskSpec(
        ...     name="my_task", build_loaders=my_loaders, build_test_loader=my_test_loader,
        ...     build_model=MyModel, batch_size=64, lr=0.01, lr_range=(1e-4, 1e-1),
        ...     data_paths=("my_task",), prepare_data=download_my_task,
        ... ))
    """
    name: str
    build_loaders: Callable[[int, int], Tuple[DataLoader, DataLoader]]
    build_test_loader: Callable[[int], DataLoader]
    build_model: Callable[[], nn.Module]
    batch_size: int
    lr: float                        # default (initial) learning rate
    lr_range: Tuple[float, float]    # search range shared by every tuning method (log scale)
    data_paths: Tuple[str, ...] = ()                # files/folders under the data directory it reads
    prepare_data: Optional[Callable[[], None]] = None  # download them locally if missing
