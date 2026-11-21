import pytest
import torch

from autotunenet.adapters.pytorch.adapter import PyTorchHyperParameterAdapter
from autotunenet.bayesian_optimizer import BayesianOptimizer
from autotunenet.config.loader import load_config_from_dict
from autotunenet.factory import build_pytorch_autotunenet
from autotunenet.metrics import DEFAULT_SMOOTHING_WINDOW, MetricSmoother
from autotunenet.parameters import ParameterSpace

SPACE = {"lr": [1e-3, 1e-2]}


@pytest.mark.parametrize("section, key", [
    ("tuning", "tune_n_steps"),
    ("tuning", "warmup_epochs"),
    ("metrics", "smoothing_window"),
    ("stability", "patience"),
    ("stability", "cooldown"),
    ("search", "n_startup_trials"),
])
def test_booleans_are_not_accepted_as_integers(section, key):
    with pytest.raises(ValueError):
        load_config_from_dict({"parameter_space": SPACE, section: {key: True}})


@pytest.mark.parametrize("kwargs", [{"tune_n_steps": True}, {"warmup_epochs": True}, {"warmup_epochs": 1.5}])
def test_adapter_rejects_non_integer_counts(kwargs):
    optimizer = torch.optim.SGD(torch.nn.Linear(2, 1).parameters(), lr=0.01)
    bo = BayesianOptimizer(ParameterSpace({"lr": (1e-3, 1e-2)}), seed=0)
    with pytest.raises(ValueError):
        PyTorchHyperParameterAdapter(optimizer, bo, **kwargs)


def test_one_default_smoothing_window_everywhere():
    optimizer = torch.optim.SGD(torch.nn.Linear(2, 1).parameters(), lr=0.01)
    from_config = build_pytorch_autotunenet(optimizer, {"parameter_space": SPACE}).autotune_optimizer
    direct = BayesianOptimizer(ParameterSpace({"lr": (1e-3, 1e-2)}))
    assert from_config.smoother.window_size == direct.smoother.window_size == DEFAULT_SMOOTHING_WINDOW
    assert MetricSmoother().window_size == DEFAULT_SMOOTHING_WINDOW
