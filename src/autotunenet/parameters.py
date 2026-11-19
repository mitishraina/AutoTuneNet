import math
import random
import warnings
from typing import Dict, Any, Tuple, List

# A range spanning at least this ratio (high / low) is wide enough that a
# linear scale badly over-samples its top end.
_WIDE_RANGE_RATIO = 100.0


class ParameterSpace:
    """
    Search space for the tuned hyperparameters.

    Each entry is one of:
        (low, high)                          continuous range, linear scale
        {"low": a, "high": b, "log": True}   continuous range; log=True samples
                                             evenly across orders of magnitude
        [v1, v2, ...]                        categorical choices

    Ranges are stored as (low, high) tuples in `space`; the scale of each range
    is in `log` (name -> bool).
    """

    def __init__(self, space: Dict[str, Any]):
        self.space: Dict[str, Any] = dict(space)
        self.log: Dict[str, bool] = {}
        self._validate_space()

    def _validate_space(self) -> None:
        for name, values in self.space.items():
            if isinstance(values, dict):
                if "low" not in values or "high" not in values:
                    raise ValueError(f"{name} range must define 'low' and 'high'")
                log = values.get("log", False)
                if not isinstance(log, bool):
                    raise ValueError(f"{name} 'log' must be true or false")
                low, high = (self._coerce_numeric(values[k], name) for k in ("low", "high"))
                self._set_range(name, low, high, log)

            elif isinstance(values, tuple):
                if len(values) != 2:
                    raise ValueError(f"{name} tuple must be (min, max)")
                low, high = (self._coerce_numeric(v, name) for v in values)
                self._set_range(name, low, high, log=False)

            elif isinstance(values, list):
                if len(values) == 0:
                    raise ValueError(f"{name} list cannot be empty")
                self.space[name] = [self._coerce_numeric(v, name) for v in values]

            else:
                raise TypeError(
                    f"Parameter {name} must be defined by a tuple, a range mapping or a list"
                )

    def _set_range(self, name: str, low: float, high: float, log: bool) -> None:
        if low > high:
            raise ValueError(f"{name}: low ({low}) must be <= high ({high})")
        if log and low <= 0:
            raise ValueError(f"{name}: a log-scale range needs low > 0 (got {low})")
        if not log and low > 0 and high / low >= _WIDE_RANGE_RATIO:
            warnings.warn(
                f"Parameter '{name}' spans {high / low:.0f}x ({low:g} to {high:g}) on a "
                f"linear scale, so most samples land near the top of the range. "
                f"Set log=True to sample evenly across orders of magnitude.",
                UserWarning,
                stacklevel=4,
            )
        self.space[name] = (low, high)
        self.log[name] = log

    def _coerce_numeric(self, value, name):
        if isinstance(value, (int, float)):
            return value

        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                raise ValueError(
                    f"Parameter '{name}' value '{value}' is not numeric"
                )

        raise TypeError(
            f"Parameter '{name}' has invalid type: {type(value)}"
        )

    def is_log(self, name: str) -> bool:
        return self.log.get(name, False)

    def sample(self) -> Dict[str, Any]:
        params = {}
        for name, values in self.space.items():
            if isinstance(values, tuple):
                low, high = values
                if self.is_log(name):
                    params[name] = math.exp(random.uniform(math.log(low), math.log(high)))
                else:
                    params[name] = random.uniform(low, high)
            elif isinstance(values, list):
                params[name] = random.choice(values)

        return params
