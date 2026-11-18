import torch
from .base import NonStationarityRegime

class GradientNoiseRegime(NonStationarityRegime):
    """
    Injects Gaussian noise into every gradient from `start_epoch` onward.

    When the regime activates it registers a gradient hook on each trainable
    parameter. The hook runs on every backward pass and adds
    N(0, std^2) noise before the optimizer step sees the gradient.
    Noise is drawn from a dedicated, seeded generator so runs are reproducible
    and the global RNG state is left untouched.
    """

    def __init__(self, start_epoch: int, std: float = 0.01, seed: int = 42):
        super().__init__(start_epoch)
        self.std = std
        self.seed = seed
        self._generator = torch.Generator().manual_seed(seed)
        self._handles = []

    def _add_noise(self, grad: torch.Tensor) -> torch.Tensor:
        noise = torch.randn(grad.shape, generator=self._generator, dtype=grad.dtype)
        return grad + noise.to(grad.device) * self.std

    def apply(self, model, **_):
        for param in model.parameters():
            if param.requires_grad:
                self._handles.append(param.register_hook(self._add_noise))

        print(
            f"[NON-STATIONARITY] Gradient noise enabled "
            f"(std={self.std}) from epoch {self.start_epoch}"
        )

    def remove(self):
        """Detach the gradient hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
