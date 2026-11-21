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
            warmup_metric_threshold: Start tuning once the loss is at or below this
                value, i.e. once metric >= -warmup_metric_threshold. Must be > 0.
            max_delta: Maximum relative change allowed per tuning step (e.g. 0.5 = +-50%, postive or negative 50% change)
        """
        if not isinstance(tune_n_steps, int) or tune_n_steps <= 0:
            raise ValueError(
                "tune_n_steps must be a positive integer "
                "(e.g. 1 = every step/epoch, 5 = every 5 steps/epochs)"
            )
            
        if warmup_epochs < 0:
            raise ValueError("warmup_epochs must be non-negative or >= 0")
        
        if warmup_metric_threshold is not None and warmup_metric_threshold <= 0:
            raise ValueError(
                "warmup_metric_threshold must be > 0: tuning starts once the loss "
                "is at or below it (metric >= -warmup_metric_threshold)"
            )

        if max_delta is not None and (max_delta <=0 or max_delta >= 1):
            raise ValueError("max_delta must be in the range (0,1)")
        
        self.torch_optimizer = torch_optimizer
        self.autotune_optimizer = autotune_optimizer
        self.tune_n_steps = tune_n_steps
        self.max_delta = max_delta
        
        self.rollback = Rollback()
        self.tracker = Tracker()
        
        self._validate_tuned_keys()

        self._epoch = 0
        self._step = 0
        # Each param group keeps its ratio to the first (reference) group, so
        # layer-wise settings such as a lower backbone LR survive tuning.
        self._group_ratios = self._compute_group_ratios()
        self._last_good_params: Dict[str, float] = self._read_current_params()
        # Fallback rollback target when nothing has been accepted yet
        # (e.g. the first tuned epoch diverges).
        self._initial_params: Dict[str, float] = dict(self._last_good_params)
        
        self.warmup_epochs = warmup_epochs
        self.warmup_metric_threshold = warmup_metric_threshold
        self._tuning_enabled = warmup_epochs == 0 and warmup_metric_threshold is None
        
        self._pending_observation = False
        
        self.last_instability = False
        self.last_rollback = False

 
    # ------------------------------------------------------------------ hooks
    # All hooks share one observe-then-suggest flow (_tuning_call). The metric
    # passed to a hook must have been measured with the hyperparameters that
    # were in effect since the previous call; it is credited to those.

    def on_epoch_end(self, metric: float):
        """Call once per epoch with a higher-is-better metric (e.g. -val_loss)."""
        self._epoch += 1
        self._tuning_call(metric)

    def on_step_end(self, metric: float):
        self._tuning_call(metric)

    def on_validation_end(self, metric: float):
        self._tuning_call(metric)

    def step(self, metric: float):
        self._tuning_call(metric)

    def step_tuning_metric(self, metric: float):
        """Deprecated alias of step(); kept for backward compatibility."""
        self._tuning_call(metric)
    
    def _warmup_satisfied(self, metric: float) -> bool:
        # Call-count warmup applies only if warmup_epochs > 0
        if self.warmup_epochs > 0 and self._step <= self.warmup_epochs:
            return False

        # Metric warmup applies only if a threshold is set
        if self.warmup_metric_threshold is not None:
            return metric >= -self.warmup_metric_threshold

        return True

    def _tuning_call(self, metric: float):
        """
        1. Score the configuration that ran since the previous tuning call
           (stability check; roll back if unstable).
        2. Propose and apply the configuration for the next interval.
        """
        self._step += 1
        self.last_instability = False
        self.last_rollback = False

        if not self._tuning_enabled:
            if not self._warmup_satisfied(metric):
                return
            # Latch: once warmup has passed it is not re-checked, so a later
            # bad epoch is scored instead of skipped.
            self._tuning_enabled = True

        if self._step % self.tune_n_steps != 0:
            return
        
        # Step 1: Observe the result of the pending configuration
        if self._pending_observation:
            current_params = self._read_current_params()
            is_stable = self.autotune_optimizer.observe(
                params=current_params,
                score=metric,
            )
            
            if not is_stable:
                self.last_instability = True
                self.last_rollback = True
                rollback_params = self.autotune_optimizer.rollback_params or self._initial_params
                if rollback_params:
                    # Restore exactly: max_delta limits how far a *proposal* may
                    # move, not how far a rollback may go back.
                    self._apply_params(rollback_params, bounded=False)
                    self.tracker.logger.info(
                        f"[ROLLBACK] Applied rollback params: {rollback_params}"
                    )
                self._pending_observation = False
                return
        
        # Step 2: Propose the configuration for the next interval
        params = self.autotune_optimizer.suggest()
        self._apply_params(params)
        self._pending_observation = True

    # ------------------------------------------------------------ helpers

    @staticmethod
    def _is_number(value) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    def _space(self):
        space = getattr(getattr(self.autotune_optimizer, "param_space", None), "space", None)
        return space if isinstance(space, dict) else None

    def _tuned_keys(self):
        """Names being tuned; every numeric setting if the search space is unknown."""
        space = self._space()
        if space is not None:
            return list(space)
        reference = self.torch_optimizer.param_groups[0]
        return [k for k, v in reference.items() if self._is_number(v)]

    def _validate_tuned_keys(self) -> None:
        """Every tuned name must be a setting of every param group."""
        space = self._space()
        if space is None:
            return
        groups = self.torch_optimizer.param_groups
        missing = [k for k in space if any(k not in g for g in groups)]
        if missing:
            available = sorted(k for k, v in groups[0].items() if self._is_number(v))
            raise ValueError(
                f"Cannot tune {missing}: not a setting of this optimizer "
                f"({type(self.torch_optimizer).__name__}). Tunable settings: {available}"
            )

    def _is_categorical(self, key) -> bool:
        space = self._space()
        return space is not None and isinstance(space.get(key), list)

    def _compute_group_ratios(self):
        groups = self.torch_optimizer.param_groups
        reference = groups[0]
        ratios = []
        for group in groups:
            r = {}
            for key in self._tuned_keys():
                if key in group and key in reference and not self._is_categorical(key):
                    ref, val = reference[key], group[key]
                    if self._is_number(ref) and self._is_number(val):
                        r[key] = val / ref if ref != 0 else 1.0
            ratios.append(r)
        return ratios

    def _step_bounds(self, key, current_value):
        """
        Allowed range for one bounded step.

        Normally current * (1 +/- max_delta). A multiplicative bound pins a
        value of 0 at 0 forever (e.g. Adam's default weight_decay), so from 0
        a step may move by max_delta * (high - low) of the search range.
        """
        if current_value != 0:
            a = current_value * (1 - self.max_delta)
            b = current_value * (1 + self.max_delta)
            return min(a, b), max(a, b)
        space = self._space()
        bounds = space.get(key) if space is not None else None
        if isinstance(bounds, tuple) and len(bounds) == 2:
            step = self.max_delta * (bounds[1] - bounds[0])
            return -step, step
        return float("-inf"), float("inf")

    def _read_current_params(self) -> Dict[str, float]:
        """Current values of the tuned hyperparameters, read from the reference group."""
        reference = self.torch_optimizer.param_groups[0]
        return {k: reference[k] for k in self._tuned_keys() if k in reference}
    
    def _apply_params(self, params: Dict[str, float], bounded: bool = True):
        """
        Write hyperparameters into the optimizer's param groups.

        Values refer to the first (reference) group; other groups get the value
        scaled by their original ratio to it. With bounded=True (proposals),
        numeric values are clipped to current * (1 +/- max_delta). Rollbacks
        pass bounded=False so the saved configuration is restored exactly.
        """
        groups = self.torch_optimizer.param_groups
        reference = groups[0]
        for key, proposed_value in params.items():
            if key not in reference:
                continue

            current_value = reference[key]
            value = proposed_value

            # Categorical values are applied exactly: clipping could produce a
            # value that is not one of the allowed choices.
            if (bounded and self.max_delta is not None and not self._is_categorical(key)
                    and self._is_number(current_value) and self._is_number(proposed_value)):
                lower, upper = self._step_bounds(key, current_value)
                value = max(lower, min(upper, proposed_value))

            for index, group in enumerate(groups):
                if key not in group:
                    continue
                ratio = self._group_ratios[index].get(key, 1.0) if index < len(self._group_ratios) else 1.0
                group[key] = value * ratio if self._is_number(value) and ratio != 1.0 else value
    
                    
    @classmethod
    def from_config(cls, torch_optimizer, config_path: str, seed: int | None = None):
        """
        Build an adapter from a YAML config file.
        Uses the same schema and defaults as autotunenet.factory.build_pytorch_autotunenet.
        """
        # Local import: factory imports this module.
        from autotunenet.factory import build_pytorch_autotunenet

        config = load_config(config_path)
        return build_pytorch_autotunenet(
            torch_optimizer=torch_optimizer,
            raw_config=config.raw_config,
            seed=seed,
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