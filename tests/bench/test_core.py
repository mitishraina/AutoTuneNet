import json

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from benchmarks.core.experiment import Cell, Experiment, parse_shard, take_shard
from benchmarks.core.metrics import cumulative_regret, post_shift_metrics
from benchmarks.core.registry import METHODS, Registry, SCENARIOS, TASKS
from benchmarks.core.result import RunResult, result_path
from benchmarks.core.trainer import Trainer

SMOKE = "benchmarks/experiments/smoke.yaml"
COMPARISON = "benchmarks/experiments/comparison.yaml"


# ------------------------------------------------------------------ registry

def test_registry_rejects_duplicates_and_names_unknowns():
    reg = Registry("thing")
    reg.add("a", 1)
    with pytest.raises(ValueError, match="already registered"):
        reg.add("a", 2)
    with pytest.raises(KeyError, match="Available: a"):
        reg.get("b")


def test_built_ins_are_registered():
    assert {"fixed_lr", "cosine", "autotunenet"} <= set(METHODS.names())
    assert {"mnist", "cifar10", "toy"} <= set(TASKS.names())
    assert {"stationary", "label_noise", "gradient_noise", "bad_initial_lr"} <= set(SCENARIOS.names())


# ---------------------------------------------------------------- experiment

def test_comparison_experiment_grid():
    exp = Experiment.load(COMPARISON)
    cells = exp.cells()
    # gradient_noise is limited to MNIST
    assert not any(c.task == "cifar10" and c.scenario == "gradient_noise" for c in cells)
    per_method = len(cells) // len(exp.methods)
    assert per_method == (4 + 3) * len(exp.seeds)


def test_shards_partition_the_grid():
    cells = Experiment.load(COMPARISON).cells()
    shards = [take_shard(cells, i, 4) for i in range(1, 5)]
    assert sorted(c for s in shards for c in s) == cells
    assert max(map(len, shards)) - min(map(len, shards)) <= 1


@pytest.mark.parametrize("bad", ["0/4", "5/4", "a/b", "3"])
def test_bad_shard_specs_raise(bad):
    with pytest.raises(ValueError):
        parse_shard(bad)


def test_unknown_names_in_an_experiment_raise(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("name: x\nepochs: 1\nseeds: [1]\ntasks: [toy]\nscenarios: {stationary: {}}\nmethods: {nope: {}}\n")
    with pytest.raises(KeyError, match="Unknown method 'nope'"):
        Experiment.load(path)


# ------------------------------------------------------------------- trainer

def test_trainer_is_deterministic_per_seed():
    task = TASKS.get("toy")
    a, b = Trainer(task, 3, torch.device("cpu")), Trainer(task, 3, torch.device("cpu"))
    for pa, pb in zip(a.model.parameters(), b.model.parameters()):
        assert torch.equal(pa, pb)


def test_evaluate_averages_loss_per_sample():
    trainer = Trainer(TASKS.get("toy"), 0, torch.device("cpu"))
    X, y = torch.randn(5, 2), torch.tensor([0, 1, 2, 3, 1])
    loss, _ = trainer.evaluate(DataLoader(TensorDataset(X, y), batch_size=4))   # batches of 4 and 1
    with torch.no_grad():
        expected = torch.nn.functional.cross_entropy(trainer.model(X), y).item()
    assert loss == pytest.approx(expected, abs=1e-6)


def test_trainer_state_round_trip():
    trainer = Trainer(TASKS.get("toy"), 0, torch.device("cpu"))
    optimizer = torch.optim.Adam(trainer.model.parameters(), lr=0.01)
    snapshot = trainer.state(optimizer)
    trainer.train_epoch(optimizer)
    trainer.load_state(snapshot, optimizer)
    for p, saved in zip(trainer.model.parameters(), snapshot["model"].values()):
        assert torch.equal(p, saved)


# ------------------------------------------------------------------- metrics

def test_post_shift_metrics_against_hand_computed_values():
    m = post_shift_metrics([0.5, 0.4, 0.3, 0.6, 0.2], shift_epoch=3)
    assert m["degradation"] == pytest.approx(0.1)          # mean(0.6, 0.2) - 0.3
    assert m["loss_spike"] == pytest.approx(0.3)           # 0.3 -> 0.6
    assert m["recovery_epochs"] == 1                       # back under 0.315 one epoch later


def test_post_shift_metrics_without_a_shift():
    assert post_shift_metrics([0.5, 0.4], None)["degradation"] is None


def test_regret_sums_the_loss_difference():
    assert cumulative_regret([0.5, 0.3], [0.4, 0.4]) == pytest.approx(0.0)


# -------------------------------------------------------------------- result

def test_result_round_trip(tmp_path):
    r = RunResult(experiment="e", method="m", task="t", scenario="s", seed=1, test_loss=0.1,
                  test_accuracy=0.9, history=[{"epoch": 0, "val_loss": 0.2}], shift_epoch=None,
                  compute={"epochs_trained": 1}, environment={}, config={})
    path = result_path(tmp_path, "e", "m", "t", "s", 1)
    r.save(path)
    assert RunResult.load(path) == r
    assert not list(path.parent.glob("*.tmp"))
