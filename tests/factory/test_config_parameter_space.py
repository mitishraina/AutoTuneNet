import pytest
import torch
import yaml

from autotunenet.config.loader import load_config_from_dict
from autotunenet.config.schema import normalize_parameter_space
from autotunenet.factory import build_pytorch_autotunenet
from autotunenet.adapters.pytorch.adapter import PyTorchHyperParameterAdapter


def _torch_optimizer(lr=0.01):
    model = torch.nn.Linear(2, 1)
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)


# --- parameter space normalization -------------------------------------------

def test_two_element_list_is_continuous_range():
    space = normalize_parameter_space({"lr": [1e-4, 1e-1]})
    assert space["lr"] == (1e-4, 1e-1)


def test_yaml_scientific_notation_strings_are_coerced():
    # PyYAML reads `1e-4` (no dot) as a string
    raw = yaml.safe_load("lr: [1e-4, 1e-1]")
    assert isinstance(raw["lr"][0], str)
    space = normalize_parameter_space(raw)
    assert space["lr"] == (1e-4, 0.1)


def test_low_high_mapping_is_range():
    space = normalize_parameter_space({"lr": {"low": 0.001, "high": 0.01}})
    assert space["lr"] == (0.001, 0.01)


def test_choices_mapping_is_categorical():
    space = normalize_parameter_space({"batch_size": {"choices": [32, 64, 128]}})
    assert space["batch_size"] == [32, 64, 128]


@pytest.mark.parametrize("bad", [
    {"lr": [0.1]},                      # wrong length
    {"lr": [0.1, 0.2, 0.3]},            # ambiguous: must use choices
    {"lr": [0.1, 0.01]},                # low > high
    {"lr": ["a", "b"]},                 # non-numeric
    {"lr": {"choices": []}},            # empty choices
    {"lr": {"low": 0.1}},               # incomplete mapping
    {"lr": 0.1},                        # scalar
])
def test_invalid_parameter_specs_raise(bad):
    with pytest.raises(ValueError):
        normalize_parameter_space(bad)


def test_config_suggestions_span_continuous_range():
    adapter = build_pytorch_autotunenet(
        torch_optimizer=_torch_optimizer(),
        raw_config={"parameter_space": {"lr": [1e-4, 1e-1]}},
        seed=0,
    )
    space = adapter.autotune_optimizer.param_space.space
    assert space["lr"] == (1e-4, 1e-1)

    suggestions = set()
    for _ in range(10):
        params = adapter.autotune_optimizer.suggest()
        adapter.autotune_optimizer.observe(params, score=0.0)
        suggestions.add(params["lr"])
        assert 1e-4 <= params["lr"] <= 1e-1
    # A categorical [1e-4, 1e-1] space could only ever produce 2 distinct values
    assert len(suggestions) > 2


# --- factory applies stability settings ---------------------------------------

def test_factory_applies_stability_config():
    adapter = build_pytorch_autotunenet(
        torch_optimizer=_torch_optimizer(),
        raw_config={
            "parameter_space": {"lr": [1e-4, 1e-1]},
            "stability": {"max_regression": 0.05, "patience": 4, "cooldown": 7},
        },
    )
    guard = adapter.autotune_optimizer.guard
    assert guard.max_regression == 0.05
    assert guard.patience == 4
    assert guard.cooldown == 7


def test_factory_stability_defaults():
    adapter = build_pytorch_autotunenet(
        torch_optimizer=_torch_optimizer(),
        raw_config={"parameter_space": {"lr": [1e-4, 1e-1]}},
    )
    guard = adapter.autotune_optimizer.guard
    assert (guard.max_regression, guard.patience, guard.cooldown) == (0.2, 2, 3)


@pytest.mark.parametrize("stability", [
    {"patience": 0},
    {"cooldown": -1},
    {"max_regression": -0.1},
])
def test_invalid_stability_config_raises(stability):
    with pytest.raises(ValueError):
        load_config_from_dict({"parameter_space": {"lr": [1e-4, 1e-1]}, "stability": stability})


# --- from_config ---------------------------------------------------------------

def test_from_config_builds_adapter(tmp_path):
    cfg = tmp_path / "autotune.yaml"
    cfg.write_text(
        "parameter_space:\n"
        "  lr: [1e-4, 1e-1]\n"
        "  momentum: {low: 0.8, high: 0.99}\n"
        "tuning:\n"
        "  tune_n_steps: 2\n"
        "  warmup_epochs: 1\n"
        "  max_delta: 0.5\n"
        "metrics:\n"
        "  smoothing_window: 4\n"
        "stability:\n"
        "  max_regression: 0.1\n"
        "  patience: 3\n"
        "  cooldown: 5\n"
    )

    optimizer = _torch_optimizer()
    adapter = PyTorchHyperParameterAdapter.from_config(
        torch_optimizer=optimizer, config_path=str(cfg), seed=42
    )

    assert isinstance(adapter, PyTorchHyperParameterAdapter)
    assert adapter.torch_optimizer is optimizer
    assert adapter.tune_n_steps == 2
    assert adapter.warmup_epochs == 1
    assert adapter.max_delta == 0.5
    assert adapter.autotune_optimizer.smoother.window_size == 4
    assert adapter.autotune_optimizer.param_space.space == {
        "lr": (1e-4, 0.1),
        "momentum": (0.8, 0.99),
    }
    guard = adapter.autotune_optimizer.guard
    assert (guard.max_regression, guard.patience, guard.cooldown) == (0.1, 3, 5)


def test_default_yaml_config_loads():
    from importlib.resources import files
    path = files("autotunenet.config").joinpath("defaults.yaml")
    adapter = PyTorchHyperParameterAdapter.from_config(
        torch_optimizer=_torch_optimizer(), config_path=str(path)
    )
    assert adapter.autotune_optimizer.param_space.space["lr"] == (0.001, 0.1)
