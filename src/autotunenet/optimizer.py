"""Base class shared by AutoTuneNet optimizers, and metric-to-score conversion."""
from abc import ABC, abstractmethod
import math
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

if TYPE_CHECKING:
    from autotunenet.parameters import ParameterSpace


def to_score(value: Any) -> float:
    """
    Convert a metric to a Python float.

    Accepts int/float, NumPy scalars (e.g. np.float32) and one-element
    tensors (e.g. torch.tensor(-0.1) or -loss.detach()).

    Args:
        value: The metric to convert.

    Returns:
        The value as a float (NaN and infinity are passed through).

    Raises:
        ValueError: `value` is a bool, or is not a single number.
    """
    if isinstance(value, bool):
        raise ValueError("Score must be a numeric value, not a bool")
    try:
        return float(value)
    except (TypeError, ValueError, RuntimeError):
        raise ValueError(
            f"Score must be a single numeric value (got {type(value).__name__})"
        ) from None


class Optimizer(ABC):
    '''
    Abstract base class for optimizers that propose hyperparameters and learn from scores.

    Subclasses implement `suggest()`. Scores are higher-is-better (pass
    `-val_loss`, not `val_loss`).

    Attributes:
        param_space: The search space.
        history: `(params, score)` pairs accepted so far, in order.
    '''
    def __init__(self, param_space: "ParameterSpace"):
        '''
        Args:
            param_space: The search space, usually a `ParameterSpace`.
        '''
        self.param_space = param_space
        self.history: List[Tuple[Dict[str, Any], float]] = []
        self._step = 0

    @abstractmethod
    def suggest(self) -> Dict[str, Any]:
        '''
        Propose the next configuration to try.

        Returns:
            Hyperparameter name to value.
        '''
        pass

    def observe(self, params: Dict[str, Any], score: float) -> None:
        '''
        Record the score of a configuration.

        Args:
            params: The hyperparameters that produced the score.
            score: Higher-is-better metric (e.g. `-val_loss`); converted with `to_score`.

        Raises:
            ValueError: `score` is not a single number.
        '''
        score = to_score(score)
        
        self._step += 1
        self.history.append((params, score))
        
    def best_result(self) -> Tuple[Dict[str, Any], float]:
        """
        Return the best `(params, score)` in `history`, ignoring non-finite scores.

        Raises:
            RuntimeError: No finite observation has been recorded yet.
        """
        # Non-finite scores never count as "best": NaN compares False with
        # everything, so a NaN entry would otherwise stick as the maximum.
        finite = [item for item in self.history if math.isfinite(item[1])]
        if not finite:
            raise RuntimeError("No observations available yet")
        return max(finite, key=lambda item: item[1])

    def best_params(self) -> Dict[str, Any]:
        """Return the hyperparameters of `best_result()`."""
        return self.best_result()[0]
    
    def best_score(self) -> float:
        """Return the score of `best_result()`."""
        return self.best_result()[1]

    def num_observations(self) -> int:
        """Return the number of entries in `history`."""
        return len(self.history)