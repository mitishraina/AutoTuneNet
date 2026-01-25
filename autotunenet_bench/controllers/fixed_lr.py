from .base import BaseController

class FixedLRController(BaseController):
    def __init__(self, lr: float):
        super().__init__()
        self.lr = lr
        
    def on_epoch_end(self, metric: float):
        pass # no adaptation

        
    def apply(self, optimizer):
        for group in optimizer.param_groups:
            group["lr"] = self.lr
            
    def name(self) -> str:
        return "FixedLR"