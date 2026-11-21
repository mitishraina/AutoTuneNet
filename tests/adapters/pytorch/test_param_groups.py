import optuna
import torch

from autotunenet.parameters import ParameterSpace
from autotunenet.bayesian_optimizer import BayesianOptimizer
from autotunenet.adapters.pytorch.adapter import PyTorchHyperParameterAdapter
from autotunenet.safeguards.stability import StabilityMonitor

optuna.logging.set_verbosity(optuna.logging.ERROR)

SPACE = {"lr": {"low": 1e-4, "high": 1e-1, "log": True}}


def _adapter(optimizer, **kwargs):
    bo = BayesianOptimizer(ParameterSpace(SPACE), seed=0, smoothing_window=1)
    return bo, PyTorchHyperParameterAdapter(optimizer, bo, **kwargs)


def test_only_tuned_hyperparameters_are_recorded_and_restored():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    bo, adapter = _adapter(optimizer)
    bo.guard = StabilityMonitor(max_regression=0.1, patience=1, cooldown=0)

    for score in [-0.1, -0.1, -5.0]:       # accept one config, then roll back
        adapter.on_epoch_end(metric=score)

    assert adapter.last_rollback is True
    assert set(bo.rollback._last_good_params) == {"lr"}
    group = optimizer.param_groups[0]
    assert group["amsgrad"] is False and group["maximize"] is False
    assert group["eps"] == 1e-08


def test_layer_wise_learning_rate_ratios_are_preserved():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    optimizer = torch.optim.SGD(
        [{"params": model[0].parameters(), "lr": 0.001},   # backbone: 10x lower
         {"params": model[1].parameters(), "lr": 0.01}],
        lr=0.001,
    )
    bo, adapter = _adapter(optimizer)

    adapter.on_epoch_end(metric=-0.1)
    lrs = [g["lr"] for g in optimizer.param_groups]
    assert lrs[0] != 0.001                       # tuned
    assert abs(lrs[1] / lrs[0] - 10.0) < 1e-9    # ratio kept


def test_rollback_restores_every_group():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    optimizer = torch.optim.SGD(
        [{"params": model[0].parameters(), "lr": 0.001},
         {"params": model[1].parameters(), "lr": 0.01}],
        lr=0.001,
    )
    bo, adapter = _adapter(optimizer)
    bo.guard = StabilityMonitor(max_regression=0.1, patience=1, cooldown=0)

    adapter.on_epoch_end(metric=-0.1)            # propose A
    lrs_a = [g["lr"] for g in optimizer.param_groups]
    adapter.on_epoch_end(metric=-0.1)            # A accepted; propose B
    adapter.on_epoch_end(metric=-5.0)            # B regressed -> back to A

    assert [g["lr"] for g in optimizer.param_groups] == lrs_a
