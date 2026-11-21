import optuna
import torch

from autotunenet.parameters import ParameterSpace
from autotunenet.bayesian_optimizer import BayesianOptimizer
from autotunenet.adapters.pytorch.adapter import PyTorchHyperParameterAdapter
from autotunenet.safeguards.stability import StabilityMonitor

optuna.logging.set_verbosity(optuna.logging.ERROR)


def _setup(max_delta=0.5):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    bo = BayesianOptimizer(
        ParameterSpace({"lr": {"low": 1e-4, "high": 1e-1, "log": True}}), seed=0, smoothing_window=1
    )
    bo.guard = StabilityMonitor(max_regression=0.1, patience=1, cooldown=0)
    return optimizer, bo, PyTorchHyperParameterAdapter(optimizer, bo, max_delta=max_delta)


def test_rollback_restores_the_last_good_value_exactly():
    optimizer, bo, adapter = _setup(max_delta=0.5)
    adapter.on_epoch_end(metric=-0.10)       # propose config A
    adapter.on_epoch_end(metric=-0.10)       # A accepted as last good; propose B
    good = bo.rollback._last_good_params["lr"]

    optimizer.param_groups[0]["lr"] = good * 6   # drift far beyond max_delta
    adapter.on_epoch_end(metric=-5.0)            # regression -> rollback

    assert adapter.last_rollback is True
    assert optimizer.param_groups[0]["lr"] == good


def test_proposals_are_still_bounded():
    optimizer, bo, adapter = _setup(max_delta=0.5)
    adapter.on_epoch_end(metric=-0.10)
    assert 0.005 <= optimizer.param_groups[0]["lr"] <= 0.015
