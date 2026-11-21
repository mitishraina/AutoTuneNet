"""Decides whether a new score is safe, or whether training should roll back."""
import math
from typing import Optional

from autotunenet.logging.logger import get_logger

_logger = get_logger()

class StabilityMonitor:
    """
    Detects unstable or harmful tuning steps.

    A step *regresses* when its score is NaN/Inf, or falls more than
    `max_regression` (relative) below the best score so far. After `patience`
    consecutive regressing steps, `update()` returns `False` and the monitor
    enters rollback. For the next `cooldown` steps no rollback is triggered,
    so the controller does not oscillate; those steps are still judged, and
    `last_step_regressed` reports the verdict.

    Attributes:
        max_regression: Relative drop counted as a regression (0.2 = 20%).
        patience: Consecutive regressions before a rollback.
        cooldown: Steps with rollback suppressed after a rollback.
        last_step_regressed: Whether the latest `update()` call saw a regression,
            even if it returned `True` (within patience, or during cooldown).
            Such a step should not become the rollback target.

    Examples:
        >>> guard = StabilityMonitor(max_regression=0.2, patience=2, cooldown=3)
        >>> guard.update(score=-1.5, best_score=-1.0)   # 50% worse: 1st regression
        True
        >>> guard.update(score=-1.5, best_score=-1.0)   # 2nd: roll back
        False
    """
    
    def __init__(self, max_regression: float = 0.2, patience: int = 2, cooldown: int = 3):
        """
        Args:
            max_regression: Relative drop from the best score counted as a
                regression; 0.2 means a score more than 20% worse.
            patience: Consecutive regressing steps that trigger a rollback.
            cooldown: Steps after a rollback during which no new rollback is triggered.
        """
        self.max_regression = max_regression
        self.patience = patience
        self.cooldown = cooldown
        
        self._cooldown_remaining = 0
        self._in_rollback = False
        self.reset()
        
    def _is_score_valid(self, score: float) -> bool:
        return not (math.isnan(score) or math.isinf(score))
        
    def is_regression(self, score: float, best_score: float) -> bool:
        """
        Return whether `score` is more than `max_regression` below `best_score`.

        The drop is relative to `abs(best_score)`. Always `False` when
        `best_score` is `None`.
        """
        if best_score is None:
            return False
            
        drop = (best_score - score)/max(abs(best_score), 1e-8)
        return drop > self.max_regression
        
    def update(self, score: float, best_score: Optional[float]) -> bool:
        """
        Judge one step and update the guard's state.

        The first step (`best_score is None`) is always accepted and resets the state.

        Args:
            score: The (smoothed) score of this step; higher is better.
            best_score: The best score accepted so far, or `None`.

        Returns:
            `True` if no rollback is needed, `False` to trigger a rollback.

        After each call, `last_step_regressed` tells whether this step was a
        regression. A step can return True and still have regressed: while
        the regression count is below `patience`, or during cooldown. Such a
        step must not become the rollback target.
        """
        _logger.debug(
            f"[STABILITY] score={score:.6f}, "
            f"best={best_score}, "
            f"bad_count={self._patience}, "
            f"cooldown={self._cooldown_remaining}, "
            f"in_rollback={self._in_rollback}"
        )
        if best_score is None:
            self._patience = 0
            self._cooldown_remaining = 0
            self._in_rollback = False
            self.last_step_regressed = False
            return True

        patience = (not self._is_score_valid(score)) or self.is_regression(score, best_score)
        self.last_step_regressed = patience
        
        if self._cooldown_remaining > 0:
            # Cooldown only suppresses a second rollback; the step is still
            # judged above, so a regressing step is not recorded as good.
            self._cooldown_remaining -= 1
            return True
            
        if patience:
            self._patience += 1
        else:
            self._patience = 0
            
        if self._patience >= self.patience:
            self._enter_rollback()
            return False
        return True
    
    
    def force_rollback(self) -> None:
        """Enter rollback immediately, bypassing patience and cooldown."""
        self._enter_rollback()

    def _enter_rollback(self):
        self._in_rollback = True
        self._patience = 0
        self._cooldown_remaining = self.cooldown
        
    def exit_rollback(self):
        """Leave the rollback state (called once a step is accepted again)."""
        self._in_rollback = False
        
    @property
    def in_rollback(self) -> bool:
        """Whether the last rollback has not yet been followed by an accepted step."""
        return self._in_rollback 
        
    def reset(self) -> None:
        """Clear the regression count, cooldown and rollback state."""
        self.last_step_regressed = False
        self._patience = 0
        self._cooldown_remaining = 0
        self._in_rollback = False 
        