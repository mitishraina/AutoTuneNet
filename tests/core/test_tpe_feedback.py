import optuna
import pytest
from optuna.trial import TrialState

from autotunenet.parameters import ParameterSpace
from autotunenet.bayesian_optimizer import BayesianOptimizer
from autotunenet.safeguards.stability import StabilityMonitor

optuna.logging.set_verbosity(optuna.logging.ERROR)

SPACE = {"lr": {"low": 1e-4, "high": 1e-1, "log": True}}


def _optimizer(window=3, patience=1, cooldown=0):
    opt = BayesianOptimizer(ParameterSpace(SPACE), seed=0, smoothing_window=window)
    opt.guard = StabilityMonitor(max_regression=0.2, patience=patience, cooldown=cooldown)
    return opt


def _feed(opt, scores):
    for s in scores:
        opt.observe(opt.suggest(), s)


def _values(opt):
    return [t.value for t in opt.study.trials if t.state == TrialState.COMPLETE]


def test_tpe_is_told_each_configs_own_score_not_the_smoothed_average():
    opt = _optimizer(window=3, patience=10)
    _feed(opt, [-0.30, -0.20, -0.10])
    assert _values(opt) == [-0.30, -0.20, -0.10]
    # The stability check still works on the smoothed series.
    assert opt.history[-1][1] == pytest.approx(-0.20)


def test_rejected_config_is_reported_with_its_score():
    opt = _optimizer(window=1)
    _feed(opt, [-0.10, -5.0])
    assert opt.last_rollback is True
    assert _values(opt) == [-0.10, -5.0]
    assert all(t.state != TrialState.FAIL for t in opt.study.trials)


def test_non_finite_score_is_reported_as_the_worst_seen():
    opt = _optimizer(window=1, patience=10)
    _feed(opt, [-0.10, -0.40, float("nan")])
    assert _values(opt) == [-0.10, -0.40, -0.40]


def test_non_finite_first_score_is_closed_as_fail():
    opt = _optimizer(window=1)
    _feed(opt, [float("nan")])
    assert [t.state for t in opt.study.trials] == [TrialState.FAIL]
