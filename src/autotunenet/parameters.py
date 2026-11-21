"""Search space definition: the hyperparameters AutoTuneNet may change, and their ranges."""
import math
import random
import warnings
from typing import Dict, Any

# A range spanning at least this ratio (high / low) is wide enough that a
# linear scale badly over-samples its top end.
_WIDE_RANGE_RATIO = 100.0


class ParameterSpace:
    """
    Search space for the tuned hyperparameters.

    Each entry maps a hyperparameter name (a key of the optimizer's
    `param_groups`, such as `lr`) to one of:

    | Form | Meaning |
    |---|---|
    | `(low, high)` | continuous range, linear scale |
    | `{"low": a, "high": b, "log": True}` | continuous range; `log=True` samples evenly across orders of magnitude (`log` defaults to `False`) |
    | `[v1, v2, ...]` | categorical choices |

    Numeric strings such as `"1e-4"` are converted to floats.

    Attributes:
        space: Name to `(low, high)` tuple (ranges) or list (choices).
        log: Name to `True` if that range uses a log scale.

    Examples:
        >>> space = ParameterSpace({
        ...     "lr": {"low": 1e-4, "high": 1e-1, "log": True},
        ...     "momentum": (0.8, 0.99),
        ...     "nesterov_like": [0.0, 1.0],
        ... })
        >>> space.is_log("lr")
        True
    """

    def __init__(self, space: Dict[str, Any]):
        """
        Args:
            space: Mapping from hyperparameter name to its range or choices
                (see the class description).

        Raises:
            ValueError: A range mapping lacks `low` or `high`, `log` is not a
                bool, a tuple does not have two elements, `low > high`, a log
                range has `low <= 0`, a list is empty, or a value is a string
                that is not a number.
            TypeError: An entry is not a tuple, mapping or list, or a value is
                not numeric.

        Warns:
            UserWarning: A linear range with `low > 0` spans 100x or more;
                use `log=True` for such ranges.
        """
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
        """Return whether `name` is a log-scale range (`False` for choices and unknown names)."""
        return self.log.get(name, False)

    def sample(self) -> Dict[str, Any]:
        """
        Draw one random configuration (uniform, or log-uniform for log ranges).

        Uses Python's `random` module, so seed it with `random.seed` for
        repeatable samples. `BayesianOptimizer` does not use this method; it
        samples through Optuna.

        Returns:
            Name to sampled value, for every entry in the space.
        """
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
