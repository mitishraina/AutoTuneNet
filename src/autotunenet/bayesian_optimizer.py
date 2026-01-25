from typing import Dict, Any
import optuna
from .optimizer import Optimizer
from .metrics import MetricSmoother
from autotunenet.safeguards.stability import StabilityMonitor
from autotunenet.safeguards.rollback import Rollback
from autotunenet.logging.tracker import Tracker
from autotunenet.logging.error_handler import log_errors

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
        self.last_instability = False
        self.last_rollback = False
    
    # def reset_history(self):
    #     self.history.clear()
    #     self._step = 0

    
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
    
    # def on_regime_start(self):
    #     # Reset internal trial state
    #     self._active_trial = None

    #     # Reset Optuna study but keep sampler seed
    #     sampler = self.study.sampler
    #     self.study = optuna.create_study(
    #         direction="maximize",
    #         sampler=sampler
    #     )

    #     # Reset guard + smoother
    #     self.guard.reset()
    #     self.smoother.reset()

    #     # Reset history
    #     self.history.clear()
    #     self._step = 0
    
    @log_errors(context="BayesianOptimizer.observe")
    def observe(self, params: Dict[str, Any], score: float) -> bool:
        if self._active_trial is None:
            raise RuntimeError("No active trial to observe")
        
        self.last_instability = False
        self.last_rollback = False
        
        smoothed_score = self.smoother.add(score)
        best_score = self.best_score() if self.history else None
        is_stable = self.guard.update(smoothed_score, best_score)
        
        trial = self._active_trial
        self._active_trial = None
        
        if is_stable:
            if self.guard.in_rollback:
                self.guard.exit_rollback()
                self.tracker.log_rollback_end()
                
            super().observe(params, smoothed_score)
            self.study.tell(trial, smoothed_score)
            self.rollback.update(params)
            
            self.tracker.log_step(
                step=self._step,
                params=params,
                score=smoothed_score,
                best_score=best_score
            )
            return True
        else:
            self.last_instability = True
            self.last_rollback = True
            # self.study.tell(trial, state=optuna.trial.TrialState.FAIL)
            self.tracker.log_rollback_start()
            return False
        
        # self.study.tell(self._active_trial, score)
        # self._active_trial = None