import math

import optuna
import pytest
import torch

from autotunenet.parameters import ParameterSpace
from autotunenet.bayesian_optimizer import BayesianOptimizer
from autotunenet.adapters.pytorch.adapter import PyTorchHyperParameterAdapter
from autotunenet.safeguards.stability import StabilityMonitor

optuna.logging.set_verbosity(optuna.logging.ERROR)

SPACE = {"lr": {"low": 1e-4, "high": 1e-1, "log": True}}
NAN, INF = float("nan"), float("inf")


def _optimizer(patience=2, cooldown=3, window=3):
    opt = BayesianOptimizer(ParameterSpace(SPACE), seed=0, smoothing_window=window)
    opt.guard = StabilityMonitor(max_regression=0.2, patience=patience, cooldown=cooldown)
    return opt


def _feed(opt, scores):
    verdicts = []
    for s in scores:
        p = opt.suggest()
        verdicts.append(opt.observe(p, s))
    return verdicts


def test_nan_first_does_not_disable_the_guard():
    opt = _optimizer(cooldown=0)
    verdicts = _feed(opt, [NAN, -0.1, -0.1, -0.1, -50.0, -50.0])
    # The NaN is rejected, and the later collapse to -50 still triggers a rollback.
    assert verdicts[0] is False
    assert False in verdicts[4:]
    assert math.isfinite(opt.best_score())


def test_non_finite_score_rolls_back_without_waiting_for_patience():
    opt = _optimizer(patience=5)
    assert _feed(opt, [-0.1, -0.1, INF]) == [True, True, False]


def test_non_finite_score_rolls_back_even_during_cooldown():
    opt = _optimizer(patience=1, cooldown=5)
    verdicts = _feed(opt, [-0.1, -5.0, NAN])
    assert verdicts == [True, False, False]


def test_non_finite_scores_stay_out_of_history_and_smoother():
    opt = _optimizer()
    _feed(opt, [-0.2, NAN, -0.1])
    assert all(math.isfinite(s) for _, s in opt.history)
    assert all(math.isfinite(v) for v in opt.smoother._values)
    assert opt.best_score() == pytest.approx(-0.15)  # mean of the two finite scores


def test_best_score_ignores_non_finite_entries():
    opt = _optimizer()
    opt.history = [({"lr": 0.01}, NAN), ({"lr": 0.02}, -0.3)]
    assert opt.best_score() == -0.3


def test_adapter_restores_initial_values_when_first_epoch_diverges():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    bo = BayesianOptimizer(ParameterSpace(SPACE), seed=0, smoothing_window=1)
    adapter = PyTorchHyperParameterAdapter(optimizer, bo)

    adapter.on_epoch_end(metric=-1.0)          # proposes and applies a new lr
    assert optimizer.param_groups[0]["lr"] != 0.01
    adapter.on_epoch_end(metric=NAN)           # that lr diverged; nothing accepted yet
    assert adapter.last_rollback is True
    assert optimizer.param_groups[0]["lr"] == 0.01
