import torch

from autotunenet.bench.core.registry import SCENARIOS
from autotunenet.bench.scenarios.base import Scenario


@SCENARIOS.register("gradient_noise")
class GradientNoise(Scenario):
    """
    From `start_epoch`, Gaussian noise N(0, std^2) is added to every gradient
    on every backward pass (via gradient hooks), from a seeded generator.
    """

    def __init__(self, start_epoch: int = 5, std: float = 0.02, seed: int = 42):
        self.shift_epoch = start_epoch
        self.std = std
        self.seed = seed
        self._generator = torch.Generator().manual_seed(seed)
        self._handles = []

    def _add_noise(self, grad: torch.Tensor) -> torch.Tensor:
        noise = torch.randn(grad.shape, generator=self._generator, dtype=grad.dtype)
        return grad + noise.to(grad.device) * self.std

    def on_epoch_start(self, epoch: int, trainer) -> None:
        if epoch == self.shift_epoch:
            for param in trainer.model.parameters():
                if param.requires_grad:
                    self._handles.append(param.register_hook(self._add_noise))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
