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
        
        self._step = 0

        # Initialize rollback with the optimizer's starting parameters
        initial_params = self._read_current_params()
        self.autotune_optimizer.rollback.update(initial_params)
        
    def on_step_end(self, metric: float):
        self.step(metric)
        
    def on_epoch_end(self, metric: float):
        self.step(metric)
        
    def on_validation_end(self, metric: float):
        self.step(metric)
        
    def step(self, metric: float):
        self._step += 1
        
        if self._step % self.tune_n_steps != 0:
            return
        
        params = self.autotune_optimizer.suggest()
        
        self._apply_params(params)

        accepted = self.autotune_optimizer.observe(params, metric)

        if not accepted:
            rollback_params = self.autotune_optimizer.rollback.rollback()
            self._apply_params(rollback_params)

        
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
    def from_config(cls, torch_optimizer, config_path: str, seed: int | None = None):
        config = load_config(config_path)

        param_space = ParameterSpace(config.parameter_space)

        autotune = BayesianOptimizer(
            param_space=param_space,
            seed=seed,
            smoothing_window=config.metrics.get("smoothing_window", 5)
        )

        autotune.guard = StabilityMonitor(
            max_regression=config.stability.get("max_regression", 0.2),
            patience=config.stability.get("patience", 2),
            cooldown=config.stability.get("cooldown", 3)
        )

        return cls(
            torch_optimizer=torch_optimizer,
            autotune_optimizer=autotune,
            tune_n_steps=config.tuning.get("tune_n_steps", 1)
        )