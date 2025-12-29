from typing import Dict, Any

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
            
        if "tune_n_steps" in self.tuning:
            if self.tuning["tune_n_steps"] <= 0:
                raise ValueError("tune_n_steps must be positive or > 0")