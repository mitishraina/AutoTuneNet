import pytest
import torch

pytest.importorskip("ray")

from benchmarks.core.ledger import ComputeLedger  # noqa: E402
from benchmarks.core.method import RunContext  # noqa: E402
from benchmarks.core.registry import TASKS  # noqa: E402
from benchmarks.methods.pbt import PBT  # noqa: E402


def _run(scenario="stationary", params=None, seed=0):
    ctx = RunContext(task=TASKS.get("toy"), scenario_name=scenario, scenario_params=params or {},
                     seed=seed, device=torch.device("cpu"), epochs=6)
    ledger = ComputeLedger()
    outcome = PBT(population=4, perturbation_interval=2).run(ctx, ledger)
    return outcome, ledger


@pytest.fixture(scope="module")
def stationary_run():
    return _run()


def test_pbt_counts_population_compute_and_exploits(stationary_run):
    outcome, ledger = stationary_run
    assert ledger.epochs_trained == 4 * 6               # population x epochs
    assert ledger.max_parallel_models == 4 and ledger.trials == 4
    assert outcome.extra["exploits"] >= 1               # weak members copied strong ones
    assert len(outcome.history) == 6                    # the reported lineage has every epoch
    assert [h["epoch"] for h in outcome.history] == list(range(6))
    assert 0 <= outcome.test_accuracy <= 1


def test_pbt_members_start_from_a_fixed_initial_lr():
    outcome, _ = _run("bad_initial_lr", {"initial_lr": 0.5})
    assert outcome.history[0]["lr"] == 0.5
    assert len(outcome.history) == 6
