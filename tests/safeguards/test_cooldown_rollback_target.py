import optuna

from autotunenet.parameters import ParameterSpace
from autotunenet.bayesian_optimizer import BayesianOptimizer
from autotunenet.safeguards.stability import StabilityMonitor

optuna.logging.set_verbosity(optuna.logging.ERROR)


def _optimizer(patience, cooldown):
    opt = BayesianOptimizer(
        ParameterSpace({"lr": {"low": 1e-4, "high": 1e-1, "log": True}}), seed=0, smoothing_window=1
    )
    opt.guard = StabilityMonitor(max_regression=0.2, patience=patience, cooldown=cooldown)
    return opt


def _step(opt, score):
    params = opt.suggest()
    return params, opt.observe(params, score)


def test_bad_step_during_cooldown_does_not_become_the_rollback_target():
    opt = _optimizer(patience=1, cooldown=2)
    good, _ = _step(opt, -0.10)
    _step(opt, -5.0)                        # rollback, cooldown starts
    _, accepted = _step(opt, -9.0)          # bad, but cooldown suppresses a rollback
    assert accepted is True
    assert opt.guard.last_step_regressed is True
    assert opt.rollback.rollback() == good


def test_bad_step_within_patience_does_not_become_the_rollback_target():
    opt = _optimizer(patience=3, cooldown=0)
    good, _ = _step(opt, -0.10)
    _, accepted = _step(opt, -5.0)          # 1st regression, below patience
    assert accepted is True
    assert opt.rollback.rollback() == good


def test_good_step_during_cooldown_updates_the_rollback_target():
    opt = _optimizer(patience=1, cooldown=2)
    _step(opt, -0.10)
    _step(opt, -5.0)
    better, _ = _step(opt, -0.09)
    assert opt.guard.last_step_regressed is False
    assert opt.rollback.rollback() == better


def test_cooldown_still_suppresses_a_second_rollback():
    guard = StabilityMonitor(max_regression=0.1, patience=1, cooldown=2)
    assert guard.update(-5.0, -0.1) is False
    assert guard.update(-9.0, -0.1) is True
    assert guard.last_step_regressed is True
