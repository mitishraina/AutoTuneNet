import torch

from benchmarks.core.method import Controller, SingleRunMethod
from benchmarks.core.registry import METHODS


class _Plateau(Controller):
    def __init__(self, optimizer, factor, patience, threshold, min_lr):
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=factor, patience=patience, threshold=threshold, min_lr=min_lr
        )
        self.reductions = 0

    def on_epoch_end(self, epoch: int, val_loss: float) -> None:
        before = self.optimizer.param_groups[0]["lr"]
        self.scheduler.step(val_loss)
        self.reductions += int(self.optimizer.param_groups[0]["lr"] < before)

    def extra(self):
        return {"lr_reductions": self.reductions}


@METHODS.register("plateau")
class ReduceLROnPlateau(SingleRunMethod):
    """
    PyTorch's ReduceLROnPlateau: multiply the learning rate by `factor` once the
    validation loss has failed to improve for more than `patience` epochs in a
    row (patience=1: two stalled epochs). A reactive rule, with no search.
    PyTorch's default patience (10) never triggers in a 10-epoch run, so the
    default here is 1.
    """

    def __init__(self, factor: float = 0.5, patience: int = 1, threshold: float = 1e-4, min_lr: float = 0.0):
        self.factor, self.patience, self.threshold, self.min_lr = factor, patience, threshold, min_lr

    def make_controller(self, optimizer, ctx):
        return _Plateau(optimizer, self.factor, self.patience, self.threshold, self.min_lr)
