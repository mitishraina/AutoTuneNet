from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class TaskSpec:
    """
    A dataset + model pair with its training defaults.

    build_loaders(batch_size, seed) -> (train_loader, val_loader)
    build_test_loader(batch_size)   -> held-out test loader (reporting only)
    build_model()                   -> a freshly initialised model

    For running on hosted runtimes (e.g. Kaggle), a task can also say which
    files it reads, relative to the data directory, and how to fetch them:
    data_paths=("MNIST/raw",), prepare_data=<function that downloads them>.
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
