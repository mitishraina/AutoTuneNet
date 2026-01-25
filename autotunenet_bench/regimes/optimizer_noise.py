import torch
from .base import NonStationarityRegime

class GradientNoiseRegime(NonStationarityRegime):
    """
    Injects Gaussian noise into gradients after a given epoch.
    """
    
    def __init__(self, start_epoch: int, std: float = 0.01):
        super().__init__(start_epoch)
        self.std = std
        
    def apply(self, model, **_):
        for param in model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * self.std
                param.grad.add_(noise)
                
        print(
            f"[NON-STATIONARITY] Gradient noise injected"
            f"(std={self.std}) at epoch {self.start_epoch}"
        )