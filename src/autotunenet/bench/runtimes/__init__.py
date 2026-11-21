"""Runtimes: where an experiment's cells are trained. `machine` (this computer) is the default."""
from autotunenet.bench.runtimes.kaggle import Kaggle
from autotunenet.bench.runtimes.machine import Machine

RUNTIMES = {"machine": Machine, "kaggle": Kaggle}


def get_runtime(name: str):
    if name not in RUNTIMES:
        raise SystemExit(f"Unknown runtime '{name}'. Available: {', '.join(RUNTIMES)}")
    return RUNTIMES[name]()
