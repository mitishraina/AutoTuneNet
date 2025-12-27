from typing import Dict, Any
from .logger import logger

class Tracker:
    def __init__(self):
        self.logger = logger()
        
    def log_step(self, step: int, params: Dict[str, Any], score: float, best_score: float) -> None:
        self.logger.info(
            f"Step {step} | params = {params}"
            f"Score={score: .6f} | params = {best_score: .6f}"
        )
    
    def log_rollback(self) -> None:
        self.logger.warning("Rollback triggered due to instability")