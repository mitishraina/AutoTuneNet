"""Load and validate AutoTuneNet configurations from YAML files or dictionaries."""
import yaml
from typing import Dict, Any
from .schema import AutoTuneConfig


def load_config(path: str) -> AutoTuneConfig:
    """
    Load and validate a configuration from a YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        The validated `AutoTuneConfig`.

    Raises:
        FileNotFoundError: The file does not exist.
        ValueError: The configuration is invalid.
    """
    with open(path, 'r') as file:
        raw_config = yaml.safe_load(file)
        
    return AutoTuneConfig(raw_config)

def load_config_from_dict(raw_config: Dict[str, Any]) -> AutoTuneConfig:
    """
    Validate a configuration given as a dictionary.

    Args:
        raw_config: The configuration, with the same sections as the YAML file.

    Returns:
        The validated `AutoTuneConfig`.

    Raises:
        ValueError: The configuration is invalid.
    """
    return AutoTuneConfig(raw_config)
