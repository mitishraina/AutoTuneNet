import torch

from benchmarks.core.registry import METHODS
from benchmarks.methods.plateau import _Plateau


def _optimizer(lr=0.1):
    return torch.optim.SGD(torch.nn.Linear(2, 1).parameters(), lr=lr)


def test_plateau_halves_the_lr_after_a_stalled_epoch():
    opt = _optimizer()
    ctl = _Plateau(opt, factor=0.5, patience=1, threshold=1e-4, min_lr=0.0)
    for epoch, loss in enumerate([1.0, 0.8, 0.8, 0.8]):
        ctl.on_epoch_end(epoch, loss)
    # improves at epoch 1; stalls at 2 and 3 -> more than `patience` (1) stalled epochs -> one reduction
    assert opt.param_groups[0]["lr"] == 0.05
    assert ctl.extra() == {"lr_reductions": 1}


def test_plateau_keeps_the_lr_while_improving():
    opt = _optimizer()
    ctl = _Plateau(opt, factor=0.5, patience=1, threshold=1e-4, min_lr=0.0)
    for epoch, loss in enumerate([1.0, 0.9, 0.8, 0.7]):
        ctl.on_epoch_end(epoch, loss)
    assert opt.param_groups[0]["lr"] == 0.1


def test_plateau_is_registered():
    assert "plateau" in METHODS.names()
