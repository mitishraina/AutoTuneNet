from typing import Dict, Any, Optional

class AutoTuneConfig:
    """
    Strongly typed representation for the AutoTuneNet configuration.
    """
    
    def __init__(self, raw_config: Dict[str, Any]):
        self.raw_config = raw_config
        
        self.parameter_space = raw_config["parameter_space"]
        self.tuning = raw_config.get("tuning", {})
        self.stability = raw_config.get("stability", {})
        self.metrics = raw_config.get("metrics", {})
        
        self._validate()
        
        
    def _validate(self):
        if not isinstance(self.parameter_space, dict):
            raise ValueError("parameter_space must be a dictionary")
        
        for name, values in self.parameter_space.items():
            if not isinstance(values, (list, tuple)):
                raise ValueError(f"invalid parameter definition for {name}")
            
        tune_n_steps = self.tuning.get("tune_n_steps")
        if tune_n_steps is not None:
            if not isinstance(tune_n_steps, int) or tune_n_steps <= 0:
                raise ValueError("tune_n_steps must be a positive integer")
            
        warmup_epochs = self.tuning.get("warmup_epochs")
        if warmup_epochs is not None:
            if not isinstance(warmup_epochs, int) or warmup_epochs < 0:
                raise ValueError("warmup_epochs must be a non-negative integer")
            
        warmup_metric_threshold = self.tuning.get("warmup_metric_threshold")
        if warmup_metric_threshold is not None:
            if not isinstance(warmup_metric_threshold, (int, float)):
                raise ValueError("warmup_metric_threshold must be a number")
            
        max_delta = self.tuning.get("max_delta")
        if max_delta is not None:
            if not isinstance(max_delta, (int, float)):
                raise ValueError("max_delta must be a number")
            if not (0.0 < max_delta < 1.0):
                raise ValueError("max_delta must be in range (0, 1)")