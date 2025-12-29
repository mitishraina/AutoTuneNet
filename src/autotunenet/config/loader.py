import yaml
from typing import Dict, Any
from .schema import AutoTuneConfig


def load_config(path: str) -> AutoTuneConfig:
    """
    Load AutoTuneNet configuration from a YAML file.
    """
    with open(path, 'r') as file:
        raw_config = yaml.safe_load(file)
        
    return AutoTuneConfig(raw_config)

def load_config_from_dict(raw_config: Dict[str, Any]) -> AutoTuneConfig:
    """
    Load AutoTuneNet configuration from a dict.
    """
    return AutoTuneConfig(raw_config)
