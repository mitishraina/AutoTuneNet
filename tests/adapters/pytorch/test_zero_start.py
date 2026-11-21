import optuna
import torch

from autotunenet.parameters import ParameterSpace
from autotunenet.bayesian_optimizer import BayesianOptimizer
from autotunenet.adapters.pytorch.adapter import PyTorchHyperParameterAdapter

optuna.logging.set_verbosity(optuna.logging.ERROR)


def test_setting_that_starts_at_zero_can_move():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)   # weight_decay = 0
    space = {"weight_decay": {"low": 1e-4, "high": 5e-4, "log": True}}
    bo = BayesianOptimizer(ParameterSpace(space), seed=0, smoothing_window=1)
    adapter = PyTorchHyperParameterAdapter(optimizer, bo, max_delta=0.5)

    adapter.on_epoch_end(metric=-0.1)
    wd = optimizer.param_groups[0]["weight_decay"]
    assert 0 < wd <= 0.5 * (5e-4 - 1e-4) + 1e-12          # first step from 0 is bounded

    for score in [-0.1, -0.1, -0.1]:
        adapter.on_epoch_end(metric=score)
    assert optimizer.param_groups[0]["weight_decay"] > 0


def test_nonzero_values_keep_the_multiplicative_bound():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    bo = BayesianOptimizer(ParameterSpace({"lr": {"low": 1e-4, "high": 1e-1, "log": True}}), seed=0)
    adapter = PyTorchHyperParameterAdapter(optimizer, bo, max_delta=0.5)
    adapter.on_epoch_end(metric=-0.1)
    assert 0.005 <= optimizer.param_groups[0]["lr"] <= 0.015
