from typing import Dict
import torch

from src.core.bayesian_optimizer import BayesianOptimizer
from src.safeguards.rollback import Rollback
from src.logging.tracker import Tracker

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
        
    def step(self, metric: float):
        self._step += 1
        
        if self._step % self.tune_n_steps != 0:
            return
        
        params = self.autotune_optimizer.suggest()
        
        old_params = self._read_current_params()
        self._apply_params(params)

        accepted = self.autotune_optimizer.observe(params, metric)
        best = self.autotune_optimizer.best_score()

        if best is not None and metric < best:
            self._apply_params(old_params)
        else:
            self._last_good_params = params.copy()


        
    def _apply_params(self, params: Dict[str, float]):
        for param_group in self.torch_optimizer.param_groups:
            if "lr" in params:
                param_group['lr'] = params["lr"]
        
        self._last_params = params.copy()
        
    # read current hyperparameters from the PyTorch optimizer
    def _read_current_params(self) -> Dict[str, float]:
        params = {}
        for group in self.torch_optimizer.param_groups:
            if "lr" in group:
                params["lr"] = group["lr"]
        return params
    
    # apply new hyperparameters to the PyTorch optimizer
    def _apply_params(self, params: Dict[str, float]):
        for group in self.torch_optimizer.param_groups:
            if "lr" in params:
                group['lr'] = params["lr"]