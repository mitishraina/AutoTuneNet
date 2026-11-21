"""The one training loop every benchmark method uses."""
import copy
import random
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Trainer:
    """
    Owns one model and its data for one run.

    Construction is seeded so that, for a given task and seed, every method
    starts from the same data split, the same initial weights and the same
    shuffling order.
    """

    def __init__(self, task, seed: int, device: torch.device, batch_size: Optional[int] = None):
        self.task = task
        self.seed = seed
        self.device = device
        self.batch_size = batch_size or task.batch_size

        seed_everything(seed)
        self.train_loader, self.val_loader = task.build_loaders(self.batch_size, seed)
        self._test_loader = None

        seed_everything(seed)
        self.model = task.build_model().to(device)

    @property
    def train_dataset(self):
        return self.train_loader.dataset

    def train_epoch(self, optimizer: torch.optim.Optimizer) -> float:
        """One pass over the training split; returns the per-sample mean loss."""
        self.model.train()
        total_loss, total = 0.0, 0
        for data, target in self.train_loader:
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            loss = F.cross_entropy(self.model(data), target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * target.size(0)
            total += target.size(0)
        return total_loss / total

    @torch.no_grad()
    def evaluate(self, loader=None) -> Tuple[float, float]:
        """(per-sample mean loss, accuracy) on the validation split by default."""
        loader = loader if loader is not None else self.val_loader
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        for data, target in loader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            total_loss += F.cross_entropy(output, target, reduction="sum").item()
            correct += (output.argmax(dim=1) == target).sum().item()
            total += target.size(0)
        return total_loss / total, correct / total

    def test(self) -> Tuple[float, float]:
        """Held-out test split: for reporting only, never for tuning or selection."""
        if self._test_loader is None:
            self._test_loader = self.task.build_test_loader(self.batch_size)
        return self.evaluate(self._test_loader)

    def state(self, optimizer: torch.optim.Optimizer) -> dict:
        """Snapshot of model and optimizer (used by population-based methods)."""
        return {"model": copy.deepcopy(self.model.state_dict()),
                "optimizer": copy.deepcopy(optimizer.state_dict())}

    def load_state(self, state: dict, optimizer: torch.optim.Optimizer) -> None:
        self.model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
