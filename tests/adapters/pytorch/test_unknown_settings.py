import optuna
import pytest
import torch
import yaml

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


@pytest.mark.parametrize("config", [
    "autotunenet_bench/configs/mnist_autotunenet.yaml",
    "autotunenet_bench/configs/mnist_gradient_noise_autotunenet.yaml",
    "autotunenet_bench/configs/cifar_autotunenet.yaml",
])
def test_benchmark_configs_only_tune_adam_settings(config):
    with open(config) as f:
        space = yaml.safe_load(f)["controller"]["parameter_space"]
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    build_pytorch_autotunenet(optimizer, {"parameter_space": space})
