import optuna
import torch.nn as nn
import torch.optim as optim
from optuna.trial import TrialState

from autotunenet.parameters import ParameterSpace
from autotunenet.bayesian_optimizer import BayesianOptimizer
from autotunenet.adapters.pytorch.adapter import PyTorchHyperParameterAdapter
from autotunenet.safeguards.stability import StabilityMonitor

optuna.logging.set_verbosity(optuna.logging.ERROR)


def _setup(**adapter_kwargs):
    model = nn.Linear(2, 1)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    space = ParameterSpace({"lr": {"low": 1e-3, "high": 1e-1, "log": True}})
    autotune = BayesianOptimizer(space, seed=42, smoothing_window=1)
    autotune.guard = StabilityMonitor(max_regression=0.2, patience=1, cooldown=0)
    adapter = PyTorchHyperParameterAdapter(optimizer, autotune, **adapter_kwargs)
    return optimizer, autotune, adapter


def _completed(study):
    return [t for t in study.trials if t.state == TrialState.COMPLETE]


def test_step_hook_rolls_back_a_regressing_config():
    optimizer, autotune, adapter = _setup(tune_n_steps=1)

    adapter.step(metric=-0.10)                 # propose config A
    lr_a = optimizer.param_groups[0]["lr"]
    adapter.step(metric=-0.10)                 # A scored and accepted; propose B
    adapter.step(metric=-5.00)                 # B regressed -> roll back to A

    assert adapter.last_rollback is True
    assert optimizer.param_groups[0]["lr"] == lr_a


def test_step_hook_credits_the_metric_to_the_config_that_produced_it():
    optimizer, autotune, adapter = _setup(tune_n_steps=1)

    lr_before = optimizer.param_groups[0]["lr"]
    adapter.step(metric=-0.123)                # measured with the initial lr: nothing pending
    assert _completed(autotune.study) == []
    lr_a = optimizer.param_groups[0]["lr"]
    assert lr_a != lr_before

    adapter.step(metric=-0.456)                # measured with config A
    (trial,) = _completed(autotune.study)
    assert trial.params["lr"] == lr_a
    assert trial.value == -0.456


def test_all_hooks_share_the_same_flow():
    results = []
    for hook in ("on_epoch_end", "on_step_end", "on_validation_end", "step"):
        optimizer, autotune, adapter = _setup()
        for m in (-0.1, -0.2, -0.3):
            getattr(adapter, hook)(metric=m)
        results.append([(t.params["lr"], t.value) for t in _completed(autotune.study)])
    assert all(r == results[0] for r in results)


def test_metric_warmup_is_latched():
    optimizer, autotune, adapter = _setup(warmup_metric_threshold=0.5)

    adapter.on_epoch_end(metric=-0.4)          # passes threshold -> propose A
    lr_a = optimizer.param_groups[0]["lr"]
    adapter.on_epoch_end(metric=-0.9)          # A did badly: scored, not skipped

    scored = [(t.params["lr"], t.value) for t in _completed(autotune.study)]
    assert scored == [(lr_a, -0.9)]
