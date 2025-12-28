from typing import Dict, Any
from ..src.autotunenet.optimizer import Optimizer

class DummyOptimizer(Optimizer):
    def suggest(self) -> Dict[str, Any]:
        return self.param_space.sample()
