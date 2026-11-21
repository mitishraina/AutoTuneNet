"""Every registered method under every scenario, on the toy task, through the real CLI."""
import pytest

from benchmarks import aggregate, run
from benchmarks.core.experiment import Experiment
from benchmarks.core.registry import METHODS, SCENARIOS
from benchmarks.core.result import RunResult

SMOKE = "benchmarks/experiments/smoke.yaml"


@pytest.fixture(scope="module")
def smoke_results(tmp_path_factory):
    root = tmp_path_factory.mktemp("results")
    assert run.main(["--experiment", SMOKE, "--results-dir", str(root), "--device", "cpu"]) == 0
    return root


def test_smoke_experiment_covers_every_registered_method_and_scenario():
    exp = Experiment.load(SMOKE)
    assert set(exp.methods) == set(METHODS.names()), "add new methods to smoke.yaml"
    assert set(exp.scenarios) == set(SCENARIOS.names()), "add new scenarios to smoke.yaml"


def test_every_cell_produces_a_valid_result(smoke_results):
    exp = Experiment.load(SMOKE)
    for cell in exp.cells():
        r = RunResult.load(smoke_results / "smoke" / cell.method / cell.task / cell.scenario / f"seed_{cell.seed}.json")
        assert 0.0 <= r.test_accuracy <= 1.0
        assert len(r.history) == exp.epochs
        assert r.compute["epochs_trained"] >= exp.epochs
        assert r.environment["device"] == "cpu"


def test_finished_cells_are_skipped(smoke_results, capsys):
    assert run.main(["--experiment", SMOKE, "--results-dir", str(smoke_results), "--device", "cpu"]) == 0
    out = capsys.readouterr().out
    assert "run:" not in out and out.count("skip (done)") == len(Experiment.load(SMOKE).cells())


def test_aggregate_summarises_the_smoke_run(smoke_results):
    aggregate.main(["--experiment", "smoke", "--results-dir", str(smoke_results)])
    import json
    summary = json.load(open(smoke_results / "smoke" / "summary.json"))
    groups = {(c.method, c.task, c.scenario) for c in Experiment.load(SMOKE).cells()}
    assert len(summary["groups"]) == len(groups)
    assert summary["warnings"] == []
