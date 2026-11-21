import numpy as np
import optuna
import pytest
import torch
from optuna.trial import TrialState

from autotunenet.parameters import ParameterSpace
from autotunenet.bayesian_optimizer import BayesianOptimizer
from autotunenet.adapters.pytorch.adapter import PyTorchHyperParameterAdapter
from autotunenet.optimizer import to_score

optuna.logging.set_verbosity(optuna.logging.ERROR)


@pytest.mark.parametrize("value", [
    -0.1, -1, np.float32(-0.1), np.float64(-0.1), torch.tensor(-0.1), torch.tensor([-0.1]),
])
def test_float_like_metrics_are_accepted(value):
    assert to_score(value) == pytest.approx(-0.1 if value != -1 else -1.0, rel=1e-6)


@pytest.mark.parametrize("value", [True, "low", None, torch.tensor([0.1, 0.2]), [0.1]])
def test_non_numeric_metrics_are_rejected(value):
    with pytest.raises(ValueError):
        to_score(value)


def test_adapter_accepts_a_tensor_metric():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    bo = BayesianOptimizer(ParameterSpace({"lr": {"low": 1e-4, "high": 1e-1, "log": True}}), seed=0)
    adapter = PyTorchHyperParameterAdapter(optimizer, bo)

    val_loss = torch.tensor(0.25)
    adapter.on_epoch_end(metric=-val_loss)
    adapter.on_epoch_end(metric=-val_loss)
    completed = [t for t in bo.study.trials if t.state == TrialState.COMPLETE]
    assert completed[0].value == pytest.approx(-0.25)
    assert isinstance(completed[0].value, float)
