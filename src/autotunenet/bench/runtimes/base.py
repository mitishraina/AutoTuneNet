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
    """
    A place where cells are trained, such as this machine or Kaggle.

    Each method is one stage of `autotunenet run` and one CLI command. Only
    `run` is required; the other stages default to doing nothing. Register a
    new runtime in `autotunenet.bench.runtimes.RUNTIMES`.

    Attributes:
        name: The value of `--runtime` that selects it.
    """
    name: str = ""

    def setup(self) -> List[str]:
        """
        Check what this runtime needs.

        Returns:
            Problems found, each a sentence saying what to fix; empty when ready.
        """
        return []

    def data(self, tasks: Sequence, force: bool = False) -> None:
        """
        Make each task's data available to this runtime.

        Args:
            tasks: `TaskSpec`s whose data is needed.
            force: Upload or fetch again even if already available.
        """

    def prepare(self, exp: Experiment, exp_path: Path, cells: Sequence[Cell], results_dir: Path,
                plugins: Sequence[str], parallel: int) -> None:
        """
        Plan the cells that have no result yet; keep a round that is still running.

        Args:
            exp: The experiment.
            exp_path: Its YAML file (shipped to hosted runtimes).
            cells: Cells to consider; those with results are skipped.
            results_dir: Root results folder.
            plugins: Plugin files to ship with the work.
            parallel: How many workers (e.g. notebooks) to use at once.
        """

    def start(self, exp: Experiment, results_dir: Path) -> None:
        """Start whatever `prepare` planned; work already started is skipped."""

    def pull(self, exp: Experiment, results_dir: Path) -> int:
        """
        Collect finished results into `results_dir` without overwriting existing ones.

        Returns:
            The number of new result files.
        """
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
        """
        Run every stage until the cells are done or the work stops.

        Args:
            exp: The experiment.
            exp_path: Its YAML file.
            cells: Cells to run; those with results are skipped.
            results_dir: Root results folder.
            plugins: Plugin files or modules.
            parallel: Workers to use at once (hosted runtimes).
            device: `auto`, `cpu`, `cuda` or `mps` (this machine).

        Returns:
            The cells that still have no result.
        """
