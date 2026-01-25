from typing import Dict
import torch

from autotunenet.bayesian_optimizer import BayesianOptimizer
from autotunenet.safeguards.rollback import Rollback
from autotunenet.logging.tracker import Tracker
from autotunenet.safeguards.stability import StabilityMonitor
from autotunenet.config.loader import load_config
from autotunenet.parameters import ParameterSpace

class PyTorchHyperParameterAdapter:
    def __init__(self, 
                 torch_optimizer: torch.optim.Optimizer,
                 autotune_optimizer: BayesianOptimizer,
                 tune_n_steps: int = 1,
                 warmup_epochs: int = 0,
                 warmup_metric_threshold: float | None = None,
                 max_delta: float | None = None):
        """
        Args:
            torch_optimizer: PyTorch Optimizer instance(eg., Adam, SGD)
            autotune_optimizer: BayesianOptimizer from AutoTuneNet
            tune_n_steps: How often to tune hyperparameters
            warmup_epochs: Number of steps/epochs before tuning starts
            warmup_metric_threshold: Minimum metric required before tuning start
            max_delta: Maximum relative change allowed per tuning step (e.g. 0.5 = +-50%, postive or negative 50% change)
        """
        if not isinstance(tune_n_steps, int) or tune_n_steps <= 0:
            raise ValueError(
                "tune_n_steps must be a positive integer "
                "(e.g. 1 = every step/epoch, 5 = every 5 steps/epochs)"
            )
            
        if warmup_epochs < 0:
            raise ValueError("warmup_epochs must be non-negative or >= 0")
        
        if max_delta is not None and (max_delta <=0 or max_delta >= 1):
            raise ValueError("max_delta must be in the range (0,1)")
        
        self.torch_optimizer = torch_optimizer
        self.autotune_optimizer = autotune_optimizer
        self.tune_n_steps = tune_n_steps
        self.max_delta = max_delta
        
        self.rollback = Rollback()
        self.tracker = Tracker()
        
        self._epoch = 0
        self._step = 0
        self._last_good_params: Dict[str, float] = self._read_current_params()
        
        self.warmup_epochs = warmup_epochs
        self.warmup_metric_threshold = warmup_metric_threshold
        self._tuning_enabled = warmup_epochs == 0 and warmup_metric_threshold is None
        
        self._pending_observation = False
        
        self.last_instability = False
        self.last_rollback = False

 
    def on_step_end(self, metric: float):
        self.step_tuning_metric(metric)
        
    def on_epoch_end(self, metric: float):
        """
        Called at the end of each epoch. Implements observe-then-suggest pattern:
        1. First observe the result of previous epoch's params (triggers stability check)
        2. Then suggest new params for next epoch
        """
        self._epoch += 1

        self.last_instability = False
        self.last_rollback = False

        if self._epoch <= self.warmup_epochs:
            return

        if (
            self.warmup_metric_threshold is not None
            and metric < self.warmup_metric_threshold
        ):
            return
        
        # Step 1: Observe the result of previous epoch's params
        if self._pending_observation:
            current_params = self._read_current_params()
            is_stable = self.autotune_optimizer.observe(
                params=current_params,
                score=metric,
            )
            
            if not is_stable:
                self.last_instability = True
                self.last_rollback = True
                # Apply rollback params
                rollback_params = self.autotune_optimizer.rollback_params
                if rollback_params:
                    self._apply_params(rollback_params)
                    self.tracker.logger.info(
                        f"[ROLLBACK] Applied rollback params: {rollback_params}"
                    )
                self._pending_observation = False
                return
        
        # Step 2: Suggest new params for next epoch
        params = self.autotune_optimizer.suggest()
        self._apply_params(params)
        self._pending_observation = True
        
        
    def on_validation_end(self, metric: float):
        self.step_tuning_metric(metric)
        
    def step(self, metric: float):
        self.step_tuning_metric(metric)
    
    def _warmup_satisfied(self, metric: float) -> bool:
        # Epoch warmup: Epoch warmup applies ONLY if warmup_epochs > 0
        if self.warmup_epochs > 0:
            if self._step <= self.warmup_epochs:
                return False

        # Metric warmup: Metric warmup applies ONLY if threshold is set
        if self.warmup_metric_threshold is not None:
            return metric >= -self.warmup_metric_threshold

        return True

        
    def step_tuning_metric(self, metric: float):
        self._step += 1

        if not self._tuning_enabled:
            if self._warmup_satisfied(metric):
                self._tuning_enabled = True
            else:
                return
        
        
        if self._step % self.tune_n_steps != 0:
            return
        
        params = self.autotune_optimizer.suggest()
        
        old_params = self._read_current_params()
        self._apply_params(params)

        accepted = self.autotune_optimizer.observe(params, metric)
        best = self.autotune_optimizer.best_score()

        if best is not None and metric < best:
            self._apply_params(old_params)
            self.tracker.log_rollback_start()
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
            for key, proposed_value in params.items():
                if key not in group:
                    continue
                
                current_value = group[key]
                
                if (self.max_delta is not None and isinstance(current_value,(int, float)) and isinstance(proposed_value,(int, float))):
                    lower = current_value * (1 - self.max_delta)
                    upper = current_value * (1 + self.max_delta)
                    proposed_value = max(lower, min(upper, proposed_value))
                    
                group[key] = proposed_value
    
                    
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
            tune_n_steps=config.adapter.get("tune_n_steps", 1),
            warmup_epochs=config.adapter.get("warmup_epochs", 0),
            warmup_metric_threshold=config.adapter.get(
                "warmup_metric_threshold", None
            ),
            max_delta=config.adapter.get("max_delta", None)
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