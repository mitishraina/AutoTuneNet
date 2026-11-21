"""Compute accounting, kept the same way for every method."""
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass


@dataclass
class ComputeLedger:
    epochs_trained: int = 0          # every epoch of every model, including discarded trials
    max_parallel_models: int = 1     # models trained at the same time (population size for PBT)
    trials: int = 1                  # training runs started (1 for single-run methods)
    restarts: int = 0                # runs restarted from scratch
    wall_time_sec: float = 0.0
    controller_time_sec: float = 0.0  # time spent in the tuning logic itself

    def add_epochs(self, n: int = 1) -> None:
        self.epochs_trained += n

    def note_parallel(self, n: int) -> None:
        self.max_parallel_models = max(self.max_parallel_models, n)

    @contextmanager
    def time_controller(self):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.controller_time_sec += time.perf_counter() - start

    def to_dict(self) -> dict:
        return asdict(self)
