"""
Where an experiment's cells are trained.

A runtime implements the stages of `autotunenet run`. Each stage is safe to
repeat: it skips work already done, so interrupting at any point and running
the same command again continues where it stopped.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Sequence

from autotunenet.bench.core.experiment import Cell, Experiment


class Runtime(ABC):
    name: str = ""

    def setup(self) -> List[str]:
        """Check what this runtime needs; returns problems (empty when ready)."""
        return []

    def data(self, tasks: Sequence, force: bool = False) -> None:
        """Make each task's data available to this runtime."""

    def prepare(self, exp: Experiment, exp_path: Path, cells: Sequence[Cell], results_dir: Path,
                plugins: Sequence[str], parallel: int) -> None:
        """Plan the unfinished cells for this runtime."""

    def start(self, exp: Experiment, results_dir: Path) -> None:
        """Start whatever prepare planned."""

    def pull(self, exp: Experiment, results_dir: Path) -> int:
        """Collect finished results into results_dir; returns the number of new results."""
        return 0

    def in_progress(self, exp: Experiment, results_dir: Path) -> bool:
        """Whether started work is still running."""
        return False

    def describe(self, exp: Experiment, results_dir: Path) -> List[str]:
        """Status lines for `autotunenet status`."""
        return []

    @abstractmethod
    def run(self, exp: Experiment, exp_path: Path, cells: Sequence[Cell], results_dir: Path,
            plugins: Sequence[str], parallel: int, device: str) -> List[Cell]:
        """All stages; returns the cells that are still unfinished."""
