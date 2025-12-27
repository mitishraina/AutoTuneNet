from collections import deque
from typing import Deque, Optional

class MetricSmoother:
    def __init__(self, window_size: int = 5):
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