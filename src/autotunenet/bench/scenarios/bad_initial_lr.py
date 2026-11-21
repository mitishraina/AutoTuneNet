from autotunenet.bench.core.registry import SCENARIOS
from autotunenet.bench.scenarios.base import Scenario


@SCENARIOS.register("bad_initial_lr")
class BadInitialLR(Scenario):
    """Every method starts from a deliberately bad learning rate (stress test)."""
    overrides_initial_lr = True

    def __init__(self, initial_lr: float = 0.1):
        self._initial_lr = initial_lr

    def initial_lr(self, task) -> float:
        return self._initial_lr
