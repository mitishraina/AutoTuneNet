from typing import Dict, Any, Optional, Tuple, List, Union


def _to_number(value: Any, name: str) -> float:
    # PyYAML (YAML 1.1) parses scientific notation without a dot, e.g. 1e-4, as a string.
    if isinstance(value, bool):
        raise ValueError(f"Parameter '{name}': boolean is not a valid numeric bound")
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    raise ValueError(f"Parameter '{name}': '{value}' is not numeric")


def normalize_parameter_space(
    raw_space: Dict[str, Any],
) -> Dict[str, Union[Tuple[float, float], Dict[str, Any], List[Any]]]:
    """
    Convert a config-style parameter space into the form ParameterSpace expects.

    Config files (YAML/JSON) have no tuple type, so the config syntax is:
        lr: [1e-4, 1e-1]                        -> continuous range (low, high), linear
        lr: {low: 1e-4, high: 1e-1}             -> continuous range, linear
        lr: {low: 1e-4, high: 1e-1, log: true}  -> continuous range, log scale
        batch_size: {choices: [32, 64]}         -> categorical choices

    Tuples are passed through unchanged as ranges.
    """
    if not isinstance(raw_space, dict):
        raise ValueError("parameter_space must be a dictionary")

    normalized: Dict[str, Union[Tuple[float, float], Dict[str, Any], List[Any]]] = {}

    for name, spec in raw_space.items():
        log = False
        if isinstance(spec, dict):
            if "choices" in spec:
                choices = spec["choices"]
                if not isinstance(choices, (list, tuple)) or len(choices) == 0:
                    raise ValueError(f"Parameter '{name}': 'choices' must be a non-empty list")
                normalized[name] = list(choices)
                continue
            if "low" in spec and "high" in spec:
                log = spec.get("log", False)
                if not isinstance(log, bool):
                    raise ValueError(f"Parameter '{name}': 'log' must be true or false")
                spec = [spec["low"], spec["high"]]
            else:
                raise ValueError(
                    f"Parameter '{name}': mapping must define either 'choices' or both 'low' and 'high'"
                )

        if isinstance(spec, (list, tuple)):
            if len(spec) != 2:
                raise ValueError(
                    f"Parameter '{name}': a range must be [low, high]; "
                    f"use {{choices: [...]}} for categorical values"
                )
            low, high = (_to_number(v, name) for v in spec)
            if low > high:
                raise ValueError(f"Parameter '{name}': low ({low}) must be <= high ({high})")
            if log:
                if low <= 0:
                    raise ValueError(f"Parameter '{name}': a log-scale range needs low > 0")
                normalized[name] = {"low": low, "high": high, "log": True}
            else:
                normalized[name] = (low, high)
            continue

        raise ValueError(f"invalid parameter definition for {name}")

    return normalized


def _is_int(value) -> bool:
    """True for ints but not bools (YAML `true` is a bool, and bool is an int subclass)."""
    return isinstance(value, int) and not isinstance(value, bool)


class AutoTuneConfig:
    """
    Strongly typed representation for the AutoTuneNet configuration.
    """

    def __init__(self, raw_config: Dict[str, Any]):
        self.raw_config = raw_config

        self.parameter_space = normalize_parameter_space(raw_config["parameter_space"])
        self.tuning = raw_config.get("tuning", {}) or {}
        self.stability = raw_config.get("stability", {}) or {}
        self.metrics = raw_config.get("metrics", {}) or {}
        self.search = raw_config.get("search", {}) or {}

        self._validate()


    def _validate(self):
        tune_n_steps = self.tuning.get("tune_n_steps")
        if tune_n_steps is not None:
            if not _is_int(tune_n_steps) or tune_n_steps <= 0:
                raise ValueError("tune_n_steps must be a positive integer")
            
        warmup_epochs = self.tuning.get("warmup_epochs")
        if warmup_epochs is not None:
            if not _is_int(warmup_epochs) or warmup_epochs < 0:
                raise ValueError("warmup_epochs must be a non-negative integer")
            
        warmup_metric_threshold = self.tuning.get("warmup_metric_threshold")
        if warmup_metric_threshold is not None:
            if isinstance(warmup_metric_threshold, bool) or not isinstance(warmup_metric_threshold, (int, float)):
                raise ValueError("warmup_metric_threshold must be a number")
            if warmup_metric_threshold <= 0:
                raise ValueError(
                    "warmup_metric_threshold must be > 0: tuning starts once the validation "
                    "loss is at or below it (the adapter receives -val_loss as the metric)"
                )
            
        max_delta = self.tuning.get("max_delta")
        if max_delta is not None:
            if not isinstance(max_delta, (int, float)):
                raise ValueError("max_delta must be a number")
            if not (0.0 < max_delta < 1.0):
                raise ValueError("max_delta must be in range (0, 1)")

        smoothing_window = self.metrics.get("smoothing_window")
        if smoothing_window is not None:
            if not _is_int(smoothing_window) or smoothing_window <= 0:
                raise ValueError("smoothing_window must be a positive integer")

        max_regression = self.stability.get("max_regression")
        if max_regression is not None:
            if not isinstance(max_regression, (int, float)) or max_regression < 0:
                raise ValueError("max_regression must be a non-negative number")

        n_startup_trials = self.search.get("n_startup_trials")
        if n_startup_trials is not None:
            if not _is_int(n_startup_trials) or n_startup_trials < 0:
                raise ValueError("n_startup_trials must be a non-negative integer")

        patience = self.stability.get("patience")
        if patience is not None:
            if not _is_int(patience) or patience <= 0:
                raise ValueError("patience must be a positive integer")

        cooldown = self.stability.get("cooldown")
        if cooldown is not None:
            if not _is_int(cooldown) or cooldown < 0:
                raise ValueError("cooldown must be a non-negative integer")