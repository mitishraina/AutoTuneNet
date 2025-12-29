from typing import Dict
import torch

from autotunenet.bayesian_optimizer import BayesianOptimizer
from autotunenet.safeguards.rollback import Rollback
from autotunenet.logging.tracker import Tracker
from autotunenet.safeguards.stability import StabilityMonitor
from autotunenet.config.loader import load_config
from autotunenet.metrics import MetricSmoother
from autotunenet.parameters import ParameterSpace

class PyTorchHyperParameterAdapter:
    def __init__(self, 
                 torch_optimizer: torch.optim.Optimizer,
                 autotune_optimizer: BayesianOptimizer,
                 tune_n_steps: int = 1):
        """
        Args:
            torch_optimizer: PyTorch Optimizer instance(eg., Adam, SGD)
            autotune_optimizer: BayesianOptimizer from AutoTuneNet
            tune_n_steps: How often to tune hyperparameters
        """
        self.torch_optimizer = torch_optimizer
        self.autotune_optimizer = autotune_optimizer
        self.tune_n_steps = tune_n_steps
        
        self.rollback = Rollback()
        self.tracker = Tracker()
        
        self._step = 0
        self._last_good_params: Dict[str, float] = self._read_current_params()
        
    def on_step_end(self, metric: float):
        self.step_tuning_metric(metric)
        
    def on_epoch_end(self, metric: float):
        self.step_tuning_metric(metric)
        
    def on_validation_end(self, metric: float):
        self.step_tuning_metric(metric)
        
    def step(self, metric: float):
        self.step_tuning_metric(metric)
        
    def step_tuning_metric(self, metric: float):
        self._step += 1
        
        if self._step % self.tune_n_steps != 0:
            return
        
        params = self.autotune_optimizer.suggest()
        
        old_params = self._read_current_params()
        self._apply_params(params)
        
        # try:
        #     self.autotune_optimizer.observe(params, metric)
        #     self._last_good_params = self._read_current_params()
        # except Exception:
        #     self._apply_params(old_params)
        #     self.tracker.log_rollback_start()

        accepted = self.autotune_optimizer.observe(params, metric)
        best = self.autotune_optimizer.best_score()

        if best is not None and metric < best:
            self._apply_params(old_params)
        else:
            self._last_good_params = params.copy()

        
    # read current hyperparameters from the PyTorch optimizer
    def _read_current_params(self) -> Dict[str, float]:
        params = {}
        for group in self.torch_optimizer.param_groups:
            for key, value in group.items():
                if isinstance(value, (int, float)):
                    params[key] = value
        return params
    
    # apply new hyperparameters to the PyTorch optimizer
    def _apply_params(self, params: Dict[str, float]):
        for group in self.torch_optimizer.param_groups:
            for key, value in params.items():
                if key in group:
                    group[key] = value
                    
                    
@classmethod
def from_config(cls, torch_optmizer, config_path: str, seed: int | None = None):
    config = load_config(config_path)
    
    param_space = ParameterSpace(config.parameter_space)
    
    autotune = BayesianOptimizer(
        param_space=param_space,
        seed=seed,
        smmothing_window=config.metrics.get("smoothing_window", 5)
    )
    
    autotune.guard = StabilityMonitor(
        max_regression=config.stability.get("max_regression", 0.2),
        patience=config.stability.get("patience", 2),
        cooldown=config.stability.get("cooldown", 3)
    )
    
    return cls(
        torch_optimizer=torch_optmizer,
        autotune_optimizer=autotune,
        tune_n_steps=config.adapter.get("tune_n_steps", 1)
    )
    
# usage example
# adapter = PyTorchHyperParameterAdapter.from_config(
#     torch_optimizer=optimizer, 
#     config_path="config.yaml", 
#     seed=42
# )
# This will allow:
#     1. version configs
#     2. share experiments
#     3. reproduce runs
#     4. avoid code changes