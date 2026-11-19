from importlib.resources import files

import optuna
import pytest
import torch

from autotunenet.adapters.pytorch.adapter import PyTorchHyperParameterAdapter
from autotunenet.bayesian_optimizer import BayesianOptimizer
from autotunenet.config.loader import load_config_from_dict
from autotunenet.parameters import ParameterSpace

optuna.logging.set_verbosity(optuna.logging.ERROR)


def _optimizer():
    model = torch.nn.Linear(2, 1)
    return torch.optim.Adam(model.parameters(), lr=0.01)


def test_packaged_defaults_start_tuning_once_loss_is_low_enough():
    path = str(files("autotunenet.config").joinpath("defaults.yaml"))
    adapter = PyTorchHyperParameterAdapter.from_config(_optimizer(), path, seed=0)

    for loss in [2.0, 1.0, 0.8]:               # warmup epochs, loss above 0.5
        adapter.on_epoch_end(metric=-loss)
    assert len(adapter.autotune_optimizer.study.trials) == 0

    adapter.on_epoch_end(metric=-0.4)           # loss <= 0.5: tuning starts
    assert len(adapter.autotune_optimizer.study.trials) == 1


@pytest.mark.parametrize("bad", [0, -0.2, True])
def test_non_positive_threshold_rejected_in_config(bad):
    with pytest.raises(ValueError):
        load_config_from_dict({
            "parameter_space": {"lr": [1e-3, 1e-2]},
            "tuning": {"warmup_metric_threshold": bad},
        })


def test_non_positive_threshold_rejected_by_adapter():
    bo = BayesianOptimizer(ParameterSpace({"lr": (1e-3, 1e-2)}), seed=0)
    with pytest.raises(ValueError, match="warmup_metric_threshold"):
        PyTorchHyperParameterAdapter(_optimizer(), bo, warmup_metric_threshold=-0.2)
