import optuna
import pytest
import torch

from autotunenet.parameters import ParameterSpace
from autotunenet.bayesian_optimizer import BayesianOptimizer
from autotunenet.adapters.pytorch.adapter import PyTorchHyperParameterAdapter
from autotunenet.factory import build_pytorch_autotunenet

optuna.logging.set_verbosity(optuna.logging.ERROR)


def test_tuning_a_setting_the_optimizer_lacks_raises():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    bo = BayesianOptimizer(ParameterSpace({"lr": (1e-3, 1e-2), "momentum": (0.8, 0.99)}), seed=0)
    with pytest.raises(ValueError, match=r"Cannot tune \['momentum'\].*Adam"):
        PyTorchHyperParameterAdapter(optimizer, bo)


def test_error_lists_available_settings():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    with pytest.raises(ValueError, match="weight_decay"):
        build_pytorch_autotunenet(optimizer, {"parameter_space": {"momentum": [0.8, 0.99]}})


def test_sgd_momentum_is_accepted():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    build_pytorch_autotunenet(optimizer, {"parameter_space": {"momentum": [0.8, 0.99]}})


@pytest.mark.parametrize("task", ["mnist", "cifar10", "toy"])
def test_benchmark_autotunenet_config_builds_against_adam(task):
    from autotunenet.bench.core.registry import TASKS, load_plugins
    from autotunenet.bench.methods.autotunenet import AutoTuneNet

    load_plugins()
    config = AutoTuneNet().config_for(TASKS.get(task))
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    build_pytorch_autotunenet(optimizer, config)
