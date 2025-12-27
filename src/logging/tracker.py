from typing import Dict, Any
from .logger import get_logger

class Tracker:
    def __init__(self):
        self.logger = get_logger()
        self._rollback_active = False

    def log_step(
        self,
        step: int,
        params: Dict[str, Any],
        score: float,
        best_score: float | None
    ) -> None:
        best_score_str = (
            f"{best_score:.6f}" if best_score is not None else "N/A"
        )

        self.logger.info(
            f"Step {step} | "
            f"params = {params} | "
            f"score = {score:.6f} | "
            f"best = {best_score_str}"
        )

    def log_rollback_start(self):
        if not self._rollback_active:
            self.logger.warning("Instability detected. Initiating rollback to last known good configuration.")
            self._rollback_active = True
            
    def log_rollback_end(self):
        if self._rollback_active:
            self.logger.info("Rollback completed. Resuming optimization.")
            self._rollback_active = False
