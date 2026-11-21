"""
Offline hyperparameter search with Optuna: many separate training runs
(trials), each with a learning rate proposed by TPE; the model with the best
final validation loss is reported. ASHA and Hyperband additionally stop weak
trials early (Optuna pruners), so they explore more trials for the same cost.
"""
import optuna
import torch

from benchmarks.core.method import Controller, Method, RunContext, train_epochs
from benchmarks.core.registry import METHODS
from benchmarks.core.result import RunOutcome


class _ReportToTrial(Controller):
    """Reports each epoch's validation loss to the trial and stops it if the pruner says so."""

    def __init__(self, trial: optuna.Trial):
        self.trial = trial

    def on_epoch_end(self, epoch: int, val_loss: float) -> None:
        self.trial.report(val_loss, step=epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()


class OptunaSearch(Method):
    """Shared implementation; subclasses choose the pruner."""

    def __init__(self, n_trials: int = 8, n_startup_trials: int = 3):
        self.n_trials = n_trials
        self.n_startup_trials = n_startup_trials

    def make_pruner(self, ctx: RunContext) -> optuna.pruners.BasePruner:
        return optuna.pruners.NopPruner()

    def run(self, ctx: RunContext, ledger) -> RunOutcome:
        optuna.logging.set_verbosity(optuna.logging.WARNING)   # one line per trial is too noisy
        low, high = ctx.task.lr_range
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=ctx.seed, n_startup_trials=self.n_startup_trials),
            pruner=self.make_pruner(ctx),
        )
        best = {"val_loss": float("inf"), "trainer": None, "history": None, "lr": None}

        def objective(trial: optuna.Trial) -> float:
            lr = trial.suggest_float("lr", low, high, log=True)
            trainer = ctx.new_trainer()            # same split, init and shuffling for every trial
            optimizer = torch.optim.Adam(trainer.model.parameters(), lr=lr)
            ledger.trials = trial.number + 1
            history = train_epochs(ctx, ledger, trainer, optimizer, ctx.new_scenario(), _ReportToTrial(trial))
            val_loss = history[-1]["val_loss"]
            if val_loss < best["val_loss"]:
                best.update(val_loss=val_loss, trainer=trainer, history=history, lr=lr)
            return val_loss

        study.optimize(objective, n_trials=self.n_trials)
        ledger.restarts = ledger.trials - 1
        if best["trainer"] is None:
            raise RuntimeError("Every trial diverged or was pruned before finishing; nothing to report")

        # The test split is used once, for the model selected on validation loss.
        test_loss, test_accuracy = best["trainer"].test()
        states = [t.state for t in study.trials]
        return RunOutcome(test_loss, test_accuracy, best["history"], {
            "best_lr": best["lr"],
            "trials_completed": states.count(optuna.trial.TrialState.COMPLETE),
            "trials_pruned": states.count(optuna.trial.TrialState.PRUNED),
        })


@METHODS.register("optuna_tpe")
class OptunaTPE(OptunaSearch):
    """Optuna TPE over full-length trials, no early stopping."""


@METHODS.register("asha")
class ASHA(OptunaSearch):
    """TPE with asynchronous successive halving (Optuna SuccessiveHalvingPruner)."""

    def __init__(self, n_trials: int = 16, n_startup_trials: int = 3, reduction_factor: int = 3,
                 min_resource: int = 1):
        super().__init__(n_trials, n_startup_trials)
        self.reduction_factor, self.min_resource = reduction_factor, min_resource

    def make_pruner(self, ctx):
        return optuna.pruners.SuccessiveHalvingPruner(
            min_resource=self.min_resource, reduction_factor=self.reduction_factor
        )


@METHODS.register("hyperband")
class Hyperband(OptunaSearch):
    """TPE with Hyperband early stopping (Optuna HyperbandPruner)."""

    def __init__(self, n_trials: int = 16, n_startup_trials: int = 3, reduction_factor: int = 3,
                 min_resource: int = 1):
        super().__init__(n_trials, n_startup_trials)
        self.reduction_factor, self.min_resource = reduction_factor, min_resource

    def make_pruner(self, ctx):
        return optuna.pruners.HyperbandPruner(
            min_resource=self.min_resource, max_resource=ctx.epochs, reduction_factor=self.reduction_factor
        )
