"""Moving-average smoothing of the tuning metric."""
from collections import deque
from typing import Deque, Optional

# Single default used by MetricSmoother, BayesianOptimizer and the config factory.
DEFAULT_SMOOTHING_WINDOW = 3


class MetricSmoother:
    """
    Mean of the last `window_size` values added.

    Examples:
        >>> s = MetricSmoother(window_size=2)
        >>> s.add(1.0), s.add(3.0), s.add(5.0)
        (1.0, 2.0, 4.0)
    """

    def __init__(self, window_size: int = DEFAULT_SMOOTHING_WINDOW):
        """
        Args:
            window_size: Number of most recent values averaged (default 3).
        """
        self.window_size = window_size
        self._values: Deque[float] = deque(maxlen = window_size)
        
    def add(self, value: float) -> float:
        """Add a value and return the new moving average."""
        self._values.append(value)
        return self.value
    
    @property
    def value(self) -> Optional[float]:
        """The current moving average, or `None` before any value is added."""
        if not self._values:
            return None
        return sum(self._values)/len(self._values)
    
    def reset(self) -> None:
        """Forget all values."""
        self._values.clear()