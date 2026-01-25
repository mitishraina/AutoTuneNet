import yaml
from typing import Dict, Any

def load_benchmark_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)
