import math
import warnings

import optuna
import pytest

from autotunenet.parameters import ParameterSpace
from autotunenet.bayesian_optimizer import BayesianOptimizer
from autotunenet.config.schema import normalize_parameter_space

optuna.logging.set_verbosity(optuna.logging.ERROR)


def _decade_counts(values):
    """How many values fall in [1e-4,1e-3), [1e-3,1e-2), [1e-2,1e-1]."""
    counts = [0, 0, 0]
    for v in values:
        counts[min(2, int(math.floor(math.log10(v))) + 4)] += 1
    return counts


def test_log_range_samples_evenly_across_decades():
    space = ParameterSpace({"lr": {"low": 1e-4, "high": 1e-1, "log": True}})
    values = [space.sample()["lr"] for _ in range(3000)]
    assert all(1e-4 <= v <= 1e-1 for v in values)
    # Each decade should get roughly a third of the samples.
    for c in _decade_counts(values):
        assert 800 < c < 1200


def test_linear_range_still_supported_and_warns_when_wide():
    with pytest.warns(UserWarning, match="log=True"):
        space = ParameterSpace({"lr": (1e-4, 1e-1)})
    assert space.space["lr"] == (1e-4, 1e-1)
    assert space.is_log("lr") is False


def test_narrow_linear_range_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ParameterSpace({"momentum": (0.8, 0.99)})


def test_log_range_requires_positive_low():
    with pytest.raises(ValueError, match="log-scale"):
        ParameterSpace({"lr": {"low": 0.0, "high": 1e-1, "log": True}})


def test_tpe_proposals_use_log_scale():
    space = ParameterSpace({"lr": {"low": 1e-4, "high": 1e-1, "log": True}})
    opt = BayesianOptimizer(space, seed=0)
    values = []
    for _ in range(10):  # random start-up phase
        params = opt.suggest()
        values.append(params["lr"])
        opt.observe(params, -0.1)
    assert opt.study.trials[0].distributions["lr"].log is True
    # On a linear scale ~9 of 10 random proposals land above 1e-2; on a log scale ~1/3 do.
    assert sum(v >= 1e-2 for v in values) <= 6


def test_config_log_syntax_is_preserved():
    space = normalize_parameter_space({
        "lr": {"low": "1e-4", "high": "1e-1", "log": True},
        "momentum": [0.8, 0.99],
    })
    assert space["lr"] == {"low": 1e-4, "high": 0.1, "log": True}
    assert space["momentum"] == (0.8, 0.99)

    ps = ParameterSpace(space)
    assert ps.space["lr"] == (1e-4, 0.1) and ps.is_log("lr")
    assert not ps.is_log("momentum")


@pytest.mark.parametrize("bad", [
    {"lr": {"low": 1e-4, "high": 1e-1, "log": "yes"}},
    {"lr": {"low": 0, "high": 1e-1, "log": True}},
])
def test_invalid_log_config_raises(bad):
    with pytest.raises(ValueError):
        normalize_parameter_space(bad)


def test_parameter_space_does_not_mutate_input():
    raw = {"lr": ("1e-4", "1e-1")}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ParameterSpace(raw)
    assert raw == {"lr": ("1e-4", "1e-1")}
