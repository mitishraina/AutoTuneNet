from optuna.trial import TrialState

from autotunenet.factory import build_pytorch_autotunenet
from benchmarks.core.method import Controller, SingleRunMethod
from benchmarks.core.registry import METHODS


class _AutoTuneNet(Controller):
    def __init__(self, adapter):
        self.adapter = adapter
        self.rollbacks = 0

    def on_epoch_end(self, epoch: int, val_loss: float) -> None:
        self.adapter.on_epoch_end(metric=-val_loss)
        self.rollbacks += int(self.adapter.last_rollback)

    def extra(self):
        study = self.adapter.autotune_optimizer.study
        return {
            "rollbacks": self.rollbacks,
            "configs_scored": sum(t.state == TrialState.COMPLETE for t in study.trials),
        }


@METHODS.register("autotunenet")
class AutoTuneNet(SingleRunMethod):
    """
    AutoTuneNet: online Bayesian tuning of the learning rate within one run,
    with stability guard and rollback. The search range is the task's lr_range
    (log scale), shared with every other tuning method.
    """

    def __init__(self, n_startup_trials: int = 3, tune_n_steps: int = 1, warmup_epochs: int = 2,
                 max_delta: float = 0.5, smoothing_window: int = 3, max_regression: float = 0.2,
                 patience: int = 2, cooldown: int = 3):
        self.settings = {
            "search": {"n_startup_trials": n_startup_trials},
            "tuning": {"tune_n_steps": tune_n_steps, "warmup_epochs": warmup_epochs, "max_delta": max_delta},
            "metrics": {"smoothing_window": smoothing_window},
            "stability": {"max_regression": max_regression, "patience": patience, "cooldown": cooldown},
        }

    def config_for(self, task) -> dict:
        low, high = task.lr_range
        return {"parameter_space": {"lr": {"low": low, "high": high, "log": True}}, **self.settings}

    def make_controller(self, optimizer, ctx):
        adapter = build_pytorch_autotunenet(optimizer, self.config_for(ctx.task), seed=ctx.seed)
        return _AutoTuneNet(adapter)
