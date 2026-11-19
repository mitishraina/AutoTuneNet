import math

import optuna
import torch
from optuna.trial import TrialState

from autotunenet.parameters import ParameterSpace
from autotunenet.bayesian_optimizer import BayesianOptimizer
from autotunenet.adapters.pytorch.adapter import PyTorchHyperParameterAdapter
from autotunenet.safeguards.stability import StabilityMonitor

optuna.logging.set_verbosity(optuna.logging.ERROR)


def _setup(space, lr=0.01, max_delta=None, **opt_kwargs):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, **opt_kwargs)
    autotune = BayesianOptimizer(ParameterSpace(dict(space)), seed=0, smoothing_window=1)
    adapter = PyTorchHyperParameterAdapter(optimizer, autotune, max_delta=max_delta)
    return optimizer, autotune, adapter


def _completed(study):
    return [t for t in study.trials if t.state == TrialState.COMPLETE]


def test_completed_trials_hold_the_values_that_actually_ran():
    optimizer, autotune, adapter = _setup({"lr": (1e-4, 1e-1)}, max_delta=0.5)

    ran = []  # lr in effect for the epoch whose score is reported next
    for score in [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5]:
        adapter.on_epoch_end(metric=score)
        ran.append(optimizer.param_groups[0]["lr"])

    completed = _completed(autotune.study)
    assert completed, "expected at least one completed trial"
    # Each scored epoch ran with the lr applied at the previous call.
    for trial, lr in zip(completed, ran):
        assert math.isclose(trial.params["lr"], lr)


def test_clipped_proposal_is_not_recorded_as_a_success():
    optimizer, autotune, adapter = _setup({"lr": (1e-4, 1e-1)}, max_delta=0.5)

    adapter.on_epoch_end(metric=-1.0)
    proposed = autotune.study.trials[0].params["lr"]
    applied = optimizer.param_groups[0]["lr"]
    assert not math.isclose(proposed, applied), "seed 0 should produce a clipped proposal"

    adapter.on_epoch_end(metric=-0.9)
    states = [t.state for t in autotune.study.trials]
    assert states[0] == TrialState.FAIL
    assert [t.params["lr"] for t in _completed(autotune.study)] == [applied]


def test_unclipped_proposal_is_completed_without_extra_trials():
    optimizer, autotune, adapter = _setup({"lr": (1e-4, 1e-1)}, max_delta=None)

    adapter.on_epoch_end(metric=-1.0)
    adapter.on_epoch_end(metric=-0.9)

    first = autotune.study.trials[0]
    assert first.state == TrialState.COMPLETE
    assert first.value == -0.9
    assert sum(t.state == TrialState.FAIL for t in autotune.study.trials) == 0


def test_rejected_trials_are_closed_not_left_running():
    optimizer, autotune, adapter = _setup({"lr": (1e-4, 1e-1)})
    autotune.guard = StabilityMonitor(max_regression=0.1, patience=1, cooldown=0)

    for score in [-0.1, -0.1, -5.0, -5.0, -5.0, -5.0]:
        adapter.on_epoch_end(metric=score)

    states = {t.state for t in autotune.study.trials}
    # Only the most recent proposal may still be waiting for its score.
    running = [t for t in autotune.study.trials if t.state == TrialState.RUNNING]
    assert len(running) <= 1
    assert TrialState.FAIL in states


def test_applied_value_outside_search_range_is_recorded():
    # Initial lr (0.05) is above the search range, so clipping lands outside it.
    optimizer, autotune, adapter = _setup({"lr": (1e-4, 1e-2)}, lr=0.05, max_delta=0.5)

    adapter.on_epoch_end(metric=-1.0)
    applied = optimizer.param_groups[0]["lr"]
    assert applied > 1e-2

    adapter.on_epoch_end(metric=-0.9)
    completed = _completed(autotune.study)
    assert math.isclose(completed[0].params["lr"], applied)
    assert completed[0].value == -0.9


def test_categorical_values_are_applied_exactly():
    optimizer, autotune, adapter = _setup(
        {"lr": (1e-4, 1e-1), "momentum": [0.0, 0.5, 0.9]}, max_delta=0.1, momentum=0.9
    )

    for score in [-1.0, -0.9, -0.8]:
        adapter.on_epoch_end(metric=score)
        assert optimizer.param_groups[0]["momentum"] in (0.0, 0.5, 0.9)

    for trial in _completed(autotune.study):
        assert trial.params["momentum"] in (0.0, 0.5, 0.9)
