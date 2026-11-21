"""The online optimizer: Optuna TPE (ask/tell) with metric smoothing, a stability guard and rollback."""
import math
from typing import Dict, Any
import optuna
from optuna.distributions import CategoricalDistribution, FloatDistribution
from optuna.trial import TrialState
from .optimizer import Optimizer, to_score
from .parameters import ParameterSpace
from .metrics import MetricSmoother, DEFAULT_SMOOTHING_WINDOW
from autotunenet.safeguards.stability import StabilityMonitor
from autotunenet.safeguards.rollback import Rollback
from autotunenet.logging.tracker import Tracker
from autotunenet.logging.error_handler import log_errors

class BayesianOptimizer(Optimizer):
    """
    Proposes hyperparameters with Optuna's TPE sampler and guards against harmful ones.

    Each tuning step is one ask/tell pair: `suggest()` asks TPE for a
    configuration, and `observe()` reports the score it produced. In
    `observe()` the score is smoothed (moving average) and passed to the
    `StabilityMonitor`:

    - **stable**: the configuration is accepted. TPE is told its raw score,
      the smoothed score goes into `history`, and, unless the step was a
      tolerated regression, it becomes the rollback target.
    - **unstable** (a regression for `patience` steps, or a NaN/Inf score at
      once): the configuration is rejected, TPE is still told its real score
      (or the worst score seen, for NaN/Inf) so it learns to avoid that
      region, and `rollback_params` holds the last good configuration.

    Usually driven by `PyTorchHyperParameterAdapter`, which also applies the
    proposals and rollbacks to the PyTorch optimizer.

    Attributes:
        study: The underlying Optuna study (direction "maximize").
        smoother: The `MetricSmoother` applied to scores.
        guard: The `StabilityMonitor`; replace it to change its settings.
        rollback: The `Rollback` store of the last good configuration.
        last_rollback: Whether the latest `observe()` rejected its configuration.
        last_instability: Same as `last_rollback`.

    Examples:
        >>> space = ParameterSpace({"lr": {"low": 1e-4, "high": 1e-1, "log": True}})
        >>> opt = BayesianOptimizer(space, seed=0, n_startup_trials=3)
        >>> params = opt.suggest()            # {"lr": ...}
        >>> accepted = opt.observe(params, score=-0.42)   # higher is better
    """
    def __init__(
        self,
        param_space: ParameterSpace,
        seed: int | None = None,
        smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
        n_startup_trials: int = 10,
    ):
        """
        Args:
            param_space: `ParameterSpace` to search.
            seed: Seed for the TPE sampler (`None` for a random seed).
            smoothing_window: Moving-average window used by the stability check.
            n_startup_trials: Number of completed trials TPE samples at random
                before it starts using its model. Optuna's default is 10; online
                runs with few tuning steps should use a smaller value, or every
                proposal will be random.

        Raises:
            ValueError: `n_startup_trials` is not a non-negative integer.
        """
        super().__init__(param_space)

        if not isinstance(n_startup_trials, int) or isinstance(n_startup_trials, bool) or n_startup_trials < 0:
            raise ValueError("n_startup_trials must be a non-negative integer")
        
        self.study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup_trials)
        )
        
        self._active_trial = None
        
        self.smoother = MetricSmoother(window_size=smoothing_window)
        self.guard = StabilityMonitor()
        self.rollback = Rollback()
        self.tracker = Tracker()
        self.last_instability = False
        self.last_rollback = False
        self._rollback_params = None 
    
    def suggest(self) -> Dict[str, Any]:
        """
        Ask TPE for the next configuration.

        Each call starts a new trial; the next `observe()` call reports its score.

        Returns:
            Hyperparameter name to proposed value, one entry per item in the space.
        """
        self._active_trial = self.study.ask()
        params = {}
        
        for name, values in self.param_space.space.items():
            if isinstance(values, tuple):
                low, high = values
                params[name] = self._active_trial.suggest_float(
                    name, low, high, log=self.param_space.is_log(name)
                )
                
            elif isinstance(values, list):
                params[name] = self._active_trial.suggest_categorical(
                    name, values
                )
                
        return params
    
    
    @log_errors(context="BayesianOptimizer.observe")
    def observe(self, params: Dict[str, Any], score: float) -> bool:
        """
        Report the score of the configuration from the last `suggest()`.

        Args:
            params: The hyperparameters that actually ran. They may differ
                from the proposal (for example after `max_delta` clipping);
                TPE records these values.
            score: Higher-is-better metric, e.g. `-val_loss`. NaN or infinity
                triggers an immediate rollback.

        Returns:
            `True` if the configuration was accepted, `False` if it was
            rejected; then `rollback_params` holds the configuration to restore
            (or `None` if no configuration has been accepted yet).

        Raises:
            RuntimeError: There is no pending `suggest()`.
            ValueError: `score` is not a single number.
        """
        if self._active_trial is None:
            raise RuntimeError("No active trial to observe")
        
        self.last_instability = False
        self.last_rollback = False

        score = to_score(score)

        if math.isfinite(score):
            smoothed_score = self.smoother.add(score)
            best_score = self.best_score() if self.history else None
            is_stable = self.guard.update(smoothed_score, best_score)
        else:
            # A NaN/Inf score means training has diverged. Roll back at once
            # (no patience, even during cooldown) and keep the value out of the
            # smoother and history, where it would poison every later comparison.
            self.tracker.logger.warning(f"Non-finite score {score}; rolling back immediately.")
            smoothed_score = None
            self.guard.force_rollback()
            is_stable = False
        
        trial = self._active_trial
        self._active_trial = None
        
        if is_stable:
            if self.guard.in_rollback:
                self.guard.exit_rollback()
                self.tracker.log_rollback_end()
                
            # The smoothed score drives the stability check and history; TPE gets
            # the raw score, which belongs to this config alone (the smoothing
            # window also contains epochs that ran under earlier configs).
            super().observe(params, smoothed_score)
            self._record_observation(trial, params, score)
            # Only a non-regressing step becomes the rollback target. A step
            # accepted during cooldown, or one still within the patience
            # allowance, may be a bad config.
            if not getattr(self.guard, "last_step_regressed", False):
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
            # Report the rejected config with the score it actually produced, so
            # TPE learns that this region is harmful (a FAIL would be ignored).
            if math.isfinite(score):
                self._record_observation(trial, params, score)
            else:
                worst = self._worst_completed_value()
                if worst is None:
                    self.study.tell(trial, state=TrialState.FAIL)
                else:
                    self._record_observation(trial, params, worst)
            # Store rollback params for adapter to apply
            if self.rollback._last_good_params is not None:
                self._rollback_params = self.rollback.rollback()
            self.tracker.log_rollback_start()
            return False
        
    def _worst_completed_value(self):
        values = [
            t.value for t in self.study.trials
            if t.state == TrialState.COMPLETE and t.value is not None and math.isfinite(t.value)
        ]
        return min(values) if values else None

    def _record_observation(self, trial, params: Dict[str, Any], score: float) -> None:
        """
        Tell the study which hyperparameters produced `score`.

        The adapter may change a proposal before it runs (for example, clipping it
        with max_delta). The score belongs to the values that actually ran, so those
        are recorded. If nothing was changed, the proposal trial is completed as-is;
        otherwise it is closed as FAIL (ignored by TPE) and a completed trial with
        the applied values is added in its place.
        """
        applied = {
            name: params.get(name, proposed)
            for name, proposed in trial.params.items()
        }

        if applied == trial.params:
            self.study.tell(trial, score)
            return

        distributions = {}
        for name, value in applied.items():
            dist = trial.distributions[name]
            if isinstance(dist, FloatDistribution) and not (dist.low <= value <= dist.high):
                # The applied value can sit outside the search range (e.g. the
                # optimizer's initial value). Widen the range for this record only.
                dist = FloatDistribution(
                    low=min(dist.low, value), high=max(dist.high, value), log=dist.log
                )
            elif isinstance(dist, CategoricalDistribution) and value not in dist.choices:
                self.tracker.logger.warning(
                    f"Applied value {value!r} for '{name}' is not one of its choices; "
                    f"recording the proposed value instead."
                )
                applied[name] = trial.params[name]
            distributions[name] = dist

        self.study.tell(trial, state=TrialState.FAIL)
        self.study.add_trial(
            optuna.trial.create_trial(params=applied, distributions=distributions, value=score)
        )

    @property
    def rollback_params(self):
        """The last good configuration, set when `observe()` rejects a configuration; `None` until then."""
        return self._rollback_params
        