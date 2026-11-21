from collections import deque
from typing import Deque, Optional

# Single default used by MetricSmoother, BayesianOptimizer and the config factory.
DEFAULT_SMOOTHING_WINDOW = 3


class MetricSmoother:
    def __init__(self, window_size: int = DEFAULT_SMOOTHING_WINDOW):
        self.window_size = window_size
        self._values: Deque[float] = deque(maxlen = window_size)
        
    def add(self, value: float) -> float:
        self._values.append(value)
        return self.value
    
    @property
    def value(self) -> Optional[float]:
        if not self._values:
            return None
        return sum(self._values)/len(self._values)
    
    def reset(self) -> None:
        self._values.clear()