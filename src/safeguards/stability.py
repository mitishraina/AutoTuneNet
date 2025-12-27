# Decide whether a new result is safe or not.
import math
from typing import Optional

class StabilityMonitor:
    """
    Detects unstable or harmful oprimization steps
    """
    
    def __init__(self, max_regression: float = 0.2, patience: int = 2, cooldown: int = 3):
        """
        Args: 
            max_regression: Maximum allowed relative regression in performance (e.g., validation loss, 0.2 means 20% drop allowed)
            patience: Number fo bad steps allowed before rollback
            cooldown: number of steps to auto-accept after rollback
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
        if best_score is None:
            return False
            
        drop = (best_score - score)/max(abs(best_score), 1e-8)
        return drop > self.max_regression
        
    def update(self, score: float, best_score: Optional[float]) -> bool:
        """
        Update guard state.
        Returns:
            True if step is considered stable. Accept step
            False if unstable. Trigger rollback
        """
        print(
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
            return True
        
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return True
        
        patience = False
        
        if not self._is_score_valid(score):
            patience = True
        elif self.is_regression(score, best_score):
            patience = True
            
        if patience:
            self._patience += 1
        else:
            self._patience = 0
            
        if self._patience >= self.patience:
            self._enter_rollback()
            return False
        return True
    
    
    def _enter_rollback(self):
        self._in_rollback = True
        self._patience = 0
        self._cooldown_remaining = self.cooldown
        
    def exit_rollback(self):
        self._in_rollback = False
        
    @property
    def in_rollback(self) -> bool:
        return self._in_rollback 
        
    def reset(self) -> None:
        self._patience = 0
        self._cooldown_remaining = 0
        self._in_rollback = False 
        