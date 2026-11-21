from autotunenet.bench.core.method import Controller, SingleRunMethod
from autotunenet.bench.core.registry import METHODS


class _NoOp(Controller):
    def on_epoch_end(self, epoch: int, val_loss: float) -> None:
        pass


@METHODS.register("fixed_lr")
class FixedLR(SingleRunMethod):
    """No tuning: the initial learning rate is kept for the whole run."""

    def make_controller(self, optimizer, ctx):
        return _NoOp()
