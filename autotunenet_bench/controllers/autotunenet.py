from autotunenet.adapters.pytorch.adapter import PyTorchHyperParameterAdapter
from .base import BaseController


class AutoTuneNetController(BaseController):
    """
    Benchmark wrapper around the AutoTuneNet adapter.

    After each on_epoch_end it reports:
        last_trial_start  a new configuration was proposed and applied
        last_update       the optimizer's tuned values changed (proposal or rollback)
        last_rollback     the stability guard rolled back
    """

    def __init__(self, adapter: PyTorchHyperParameterAdapter):
        super().__init__()
        self.adapter = adapter
        self.last_update = False
        self.last_trial_start = False

    def _tuned_values(self):
        space = getattr(self.adapter.autotune_optimizer.param_space, "space", {})
        return tuple(
            group.get(key) for group in self.adapter.torch_optimizer.param_groups for key in space
        )

    def on_epoch_end(self, metric: float):
        # Capture state *before* the adapter acts; reading it afterwards made
        # every comparison trivially equal.
        trial_before = getattr(self.adapter.autotune_optimizer, "_active_trial", None)
        values_before = self._tuned_values()

        self.adapter.on_epoch_end(metric)

        trial_after = getattr(self.adapter.autotune_optimizer, "_active_trial", None)
        self.last_trial_start = trial_after is not None and trial_after is not trial_before
        self.last_update = self._tuned_values() != values_before
        self.last_instability = self.adapter.last_instability
        self.last_rollback = self.adapter.last_rollback

    def apply(self, optimizer):
        # adapter already applied params internally
        pass

    def name(self) -> str:
        return "AutoTuneNet"
