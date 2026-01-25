from autotunenet.adapters.pytorch.adapter import PyTorchHyperParameterAdapter
from .base import BaseController


class AutoTuneNetController(BaseController):
    def __init__(self, adapter: PyTorchHyperParameterAdapter):
        self.adapter = adapter
        self.last_instability = False
        self.last_rollback = False
        self.last_update = False
        self.last_trial_start = False
        
        
    def on_epoch_end(self, metric: float):
        self.adapter.on_epoch_end(metric)
        self.last_instability = self.adapter.last_instability
        self.last_rollback = self.adapter.last_rollback
        prev_pending = self.adapter._pending_observation
        
        if not prev_pending and self.adapter._pending_observation:
            self.last_trial_start = True
        else:
            self.last_trial_start = False
            
        self.last_update = self.adapter._pending_observation

    def apply(self, optimizer):
        # adapter already applied params internally
        pass

    def name(self) -> str:
        return "AutoTuneNet"
