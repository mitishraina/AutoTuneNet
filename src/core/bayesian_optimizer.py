from typing import Dict, Any
import optuna
from .optimizer import Optimizer
from .metrics import MetricSmoother
from src.safeguards.stability import StabilityMonitor
from src.safeguards.rollback import Rollback
from src.logging.tracker import Tracker
from src.logging.error_handler import log_errors

log

class BayesianOptimizer(Optimizer):
    """
    Bayesian optimizer using Optuna's TPE sampler.(Using ask-tell interface)
    """
    def __init__(self, param_space, seed: int | None = None, smoothing_window: int=5):
        super().__init__(param_space)
        
        self.study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed)
        )
        
        self._active_trial = None
        
        self.smoother = MetricSmoother(window_size=smoothing_window)
        self.guard = StabilityMonitor()
        self.rollback = Rollback()
        self.tracker = Tracker()
        
    def suggest(self) -> Dict[str, Any]:
        self._active_trial = self.study.ask()
        params = {}
        
        for name, values in self.param_space.space.items():
            if isinstance(values, tuple):
                low, high = values
                params[name] = self._active_trial.suggest_float(
                    name, low, high
                )
                
            elif isinstance(values, list):
                params[name] = self._active_trial.suggest_categorical(
                    name, values
                )
                
        return params
    
    @log_errors(context="BayesianOptimizer.observe")
    def observe(self, params: Dict[str, Any], score: float) -> None:
        if self._active_trial is None:
            raise RuntimeError("No active trial to observe")
        
        smoothed_score = self.smoother.add(score)
        best_score = self.best_score() if self.history else None
        is_stable = self.guard.update(smoothed_score, best_score)
        
        if is_stable:
            if self.guard.in_rollback:
                self.guard.exit_rollback()
                self.tracker.log_rollback_end()
            super().observe(params, smoothed_score)
            self.study.tell(self._active_trial, smoothed_score)
            self.rollback.update(params)
            
            self.tracker.log_step(
                step=self._step,
                params=params,
                score=smoothed_score,
                best_score=best_score
            )
        else:
            self.tracker.log_rollback_start()
        
        # self.study.tell(self._active_trial, score)
        self._active_trial = None