# Decide whether a new result is safe or not.
import math
from typing import Optional

class StabilityMonitor:
    """
    Detects unstable or harmful oprimization steps
    """
    
    def __init__(self, max_regression: float = 0.2, patience: int = 2):
        """
        Args: 
            max_regression: Maximum allowed relative regression in performance (e.g., validation loss, 0.2 means 20% drop allowed)
            patience: Number fo bad steps allowed before rollback
        """
        self.max_regression = max_regression
        self.patience = patience
        self._bad_steps = 0
        
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
                True if step is considered stable.
                False if unstable
            """
            if not self.is_score_valid(score):
                self._bad_steps += 1
            elif self.is_regression(score, best_score):
                self._bad_steps += 1
            else:
                self._bad_steps = 0
                
            return self._bad_steps <= self.patience
        
        def reset(self) -> None:
            self._bad_steps = 0
        