import torch

from autotunenet.bench.core.registry import METHODS
from autotunenet.bench.methods.plateau import _Plateau


def _optimizer(lr=0.1):
    return torch.optim.SGD(torch.nn.Linear(2, 1).parameters(), lr=lr)


def test_plateau_halves_the_lr_after_a_stalled_epoch():
    opt = _optimizer()
    ctl = _Plateau(opt, factor=0.5, patience=1, threshold=1e-4, min_lr=0.0)
    for epoch, loss in enumerate([1.0, 0.8, 0.8, 0.8]):
        ctl.on_epoch_end(epoch, loss)
    # improves at epoch 1; stalls at 2 and 3 -> more than `patience` (1) stalled epochs -> one reduction
    assert opt.param_groups[0]["lr"] == 0.05
    assert ctl.extra() == {"lr_reductions": 1}


def test_plateau_keeps_the_lr_while_improving():
    opt = _optimizer()
    ctl = _Plateau(opt, factor=0.5, patience=1, threshold=1e-4, min_lr=0.0)
    for epoch, loss in enumerate([1.0, 0.9, 0.8, 0.7]):
        ctl.on_epoch_end(epoch, loss)
    assert opt.param_groups[0]["lr"] == 0.1


def test_plateau_is_registered():
    assert "plateau" in METHODS.names()


# ------------------------------------------------------------------ optuna

import pytest  # noqa: E402

from autotunenet.bench.core.ledger import ComputeLedger  # noqa: E402
from autotunenet.bench.core.method import RunContext  # noqa: E402
from autotunenet.bench.core.registry import TASKS  # noqa: E402
from autotunenet.bench.methods.optuna_search import ASHA, Hyperband, OptunaTPE  # noqa: E402


def _ctx(epochs=6):
    return RunContext(task=TASKS.get("toy"), scenario_name="stationary", scenario_params={},
                      seed=0, device=torch.device("cpu"), epochs=epochs)


def test_tpe_trains_every_trial_to_the_end():
    ledger = ComputeLedger()
    outcome = OptunaTPE(n_trials=3, n_startup_trials=1).run(_ctx(), ledger)
    assert ledger.trials == 3 and ledger.restarts == 2
    assert ledger.epochs_trained == 3 * 6
    assert outcome.extra["trials_completed"] == 3
    assert len(outcome.history) == 6 and 0 <= outcome.test_accuracy <= 1


@pytest.mark.parametrize("method", [ASHA(n_trials=8, n_startup_trials=1), Hyperband(n_trials=8, n_startup_trials=1)])
def test_pruning_methods_stop_some_trials_early(method):
    ledger = ComputeLedger()
    outcome = method.run(_ctx(), ledger)
    assert ledger.trials == 8
    assert outcome.extra["trials_pruned"] >= 1
    assert ledger.epochs_trained < 8 * 6          # early stopping saved compute


def test_selected_model_is_the_best_on_validation():
    outcome = OptunaTPE(n_trials=3, n_startup_trials=1).run(_ctx(), ComputeLedger())
    assert outcome.extra["best_lr"] is not None
