import torch

from benchmarks.core.method import Controller, SingleRunMethod
from benchmarks.core.registry import METHODS


class _Cosine(Controller):
    def __init__(self, optimizer, epochs: int):
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    def on_epoch_end(self, epoch: int, val_loss: float) -> None:
        self.scheduler.step()


@METHODS.register("cosine")
class Cosine(SingleRunMethod):
    """Cosine annealing from the initial learning rate to 0 over the run (fixed schedule)."""

    def make_controller(self, optimizer, ctx):
        return _Cosine(optimizer, ctx.epochs)
