import math
from typing import Dict, Any
import optuna
from optuna.distributions import CategoricalDistribution, FloatDistribution
from optuna.trial import TrialState
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
    def __init__(
        self,
        param_space,
        seed: int | None = None,
        smoothing_window: int = 5,
        n_startup_trials: int = 10,
    ):
        """
        Args:
            param_space: ParameterSpace to search.
            seed: Seed for the TPE sampler.
            smoothing_window: Moving-average window used by the stability check.
            n_startup_trials: Number of completed trials TPE samples at random
                before it starts using its model. Optuna's default is 10; online
                runs with few tuning steps should use a smaller value, or every
                proposal will be random.
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
        if self._active_trial is None:
            raise RuntimeError("No active trial to observe")
        
        self.last_instability = False
        self.last_rollback = False

        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("Score must be a numeric value")

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
                
            super().observe(params, smoothed_score)
            self._record_observation(trial, params, smoothed_score)
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
            # Close the trial so it is not left RUNNING forever. FAIL trials are
            # ignored by TPE.
            self.study.tell(trial, state=TrialState.FAIL)
            # Store rollback params for adapter to apply
            if self.rollback._last_good_params is not None:
                self._rollback_params = self.rollback.rollback()
            self.tracker.log_rollback_start()
            return False
        
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
        """Returns the params to rollback to, or None if no rollback needed."""
        return self._rollback_params
        