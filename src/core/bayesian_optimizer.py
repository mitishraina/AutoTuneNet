from typing import Dict, Any
import optuna
from .optimizer import Optimizer
import logging

class BayesianOptimizer(Optimizer):
    """
    Bayesian optimizer using Optuna's TPE sampler.(Using ask-tell interface)
    """
    def __init__(self, param_space, seed: int | None = None):
        super().__init__(param_space)
        
        self.study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed)
        )
        
        self._active_trial = None
        
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
    
    def observe(self, params: Dict[str, Any], score: float) -> None:
        super().observe(params, score)
        
        if self._active_trial is None:
            logging.info("No active trial to observe")
            raise RuntimeError("No active trial to observe")
        
        self.study.tell(self._active_trial, score)
        self._active_trial = None