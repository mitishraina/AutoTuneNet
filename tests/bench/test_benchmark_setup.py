import pytest
import yaml

from autotunenet.config.schema import normalize_parameter_space
from autotunenet_bench.aggregate import _split, _stats
from benchmarks.common.device import resolve_device
from benchmarks.common.legacy import autotunenet_config


@pytest.mark.parametrize("task, config", [
    ("mnist", "autotunenet_bench/configs/mnist_autotunenet.yaml"),
    ("cifar10", "autotunenet_bench/configs/cifar_autotunenet.yaml"),
])
def test_legacy_autotunenet_settings_match_the_bench_configs(task, config):
    with open(config) as f:
        bench = yaml.safe_load(f)["controller"]
    legacy = autotunenet_config(task)

    bench_lr = normalize_parameter_space(bench["parameter_space"])["lr"]
    legacy_lr = normalize_parameter_space(legacy["parameter_space"])["lr"]
    assert legacy_lr == bench_lr
    for section in ("search", "tuning", "metrics", "stability"):
        assert legacy[section] == bench[section], section


def test_config_names_split_into_setting_and_controller():
    assert _split("mnist_gradient_noise_autotunenet") == ("mnist_gradient_noise", "autotunenet")
    assert _split("cifar_fixed_lr") == ("cifar", "fixed_lr")
    assert _split("mnist_scheduler") == ("mnist", "scheduler")


def test_stats_reports_mean_and_sample_std():
    s = _stats([0.9, 0.92, 0.94])
    assert s["mean"] == pytest.approx(0.92)
    assert s["std"] == pytest.approx(0.02)
    assert s["n"] == 3
    assert _stats([None]) is None


def test_cpu_device_always_resolves():
    assert resolve_device("cpu").type == "cpu"
