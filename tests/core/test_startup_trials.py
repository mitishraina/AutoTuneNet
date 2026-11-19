import optuna
import pytest
import torch

from autotunenet.parameters import ParameterSpace
from autotunenet.bayesian_optimizer import BayesianOptimizer
from autotunenet.factory import build_pytorch_autotunenet
from autotunenet.config.loader import load_config_from_dict

optuna.logging.set_verbosity(optuna.logging.ERROR)

SPACE = {"lr": {"low": 1e-4, "high": 1e-1, "log": True}}


def _proposals(n_startup, steps=6):
    opt = BayesianOptimizer(ParameterSpace(SPACE), seed=0, smoothing_window=1, n_startup_trials=n_startup)
    out = []
    for _ in range(steps):
        p = opt.suggest()
        out.append(p["lr"])
        opt.observe(p, -abs(p["lr"] - 0.003))
    return out


def test_model_based_proposals_start_after_startup_trials():
    random_only = _proposals(n_startup=100)
    model_based = _proposals(n_startup=2)
    # Same seed and same feedback: identical while both are sampling at random...
    assert model_based[:2] == random_only[:2]
    # ...and different once TPE starts using its model.
    assert model_based[2:] != random_only[2:]


def test_default_matches_optuna():
    opt = BayesianOptimizer(ParameterSpace(SPACE), seed=0)
    assert opt.study.sampler._n_startup_trials == 10


@pytest.mark.parametrize("bad", [-1, 1.5, True])
def test_invalid_startup_trials_rejected(bad):
    with pytest.raises(ValueError):
        BayesianOptimizer(ParameterSpace(SPACE), n_startup_trials=bad)
    with pytest.raises(ValueError):
        load_config_from_dict({"parameter_space": SPACE, "search": {"n_startup_trials": bad}})


def test_factory_passes_startup_trials():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    adapter = build_pytorch_autotunenet(
        optimizer, {"parameter_space": SPACE, "search": {"n_startup_trials": 3}}, seed=0
    )
    assert adapter.autotune_optimizer.study.sampler._n_startup_trials == 3
