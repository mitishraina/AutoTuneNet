"""Every registered method under every scenario, on the toy task, through the real CLI."""
import pytest

from autotunenet import cli
from autotunenet.bench import aggregate
from autotunenet.bench.core.experiment import Experiment
from autotunenet.bench.core.registry import METHODS, SCENARIOS
from autotunenet.bench.core.result import RunResult

SMOKE = "smoke"


@pytest.fixture(scope="module")
def smoke_results(tmp_path_factory):
    root = tmp_path_factory.mktemp("results")
    assert cli.main(["run", "--experiment", SMOKE, "--results-dir", str(root), "--device", "cpu", "--no-report"]) == 0
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
    n = len(Experiment.load(SMOKE).cells())
    assert cli.main(["run", "--experiment", SMOKE, "--results-dir", str(smoke_results), "--device", "cpu",
                     "--no-report"]) == 0
    out = capsys.readouterr().out
    assert f"0 of {n} cells to run" in out and f"all {n} cells done" in out


def test_aggregate_summarises_the_smoke_run(smoke_results):
    summary, warnings = aggregate.summarise(aggregate.load_results(smoke_results, "smoke"))
    groups = {(c.method, c.task, c.scenario) for c in Experiment.load(SMOKE).cells()}
    assert len(summary) == len(groups)
    assert warnings == []
