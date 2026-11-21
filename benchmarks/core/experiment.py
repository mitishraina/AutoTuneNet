"""Experiment files: which methods, tasks, scenarios and seeds to run."""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import yaml

from benchmarks.core.registry import METHODS, SCENARIOS, TASKS


@dataclass(frozen=True, order=True)
class Cell:
    """One run: a method on a task under a scenario with a seed."""
    task: str
    scenario: str
    method: str
    seed: int

    def label(self) -> str:
        return f"{self.method} / {self.task} / {self.scenario} / seed {self.seed}"


@dataclass
class Experiment:
    """
    Example (benchmarks/experiments/comparison.yaml):

        name: comparison
        epochs: 10
        seeds: [42, 43, 44]
        tasks: [mnist, cifar10]
        scenarios:
          stationary: {}
          gradient_noise: {start_epoch: 5, std: 0.02, tasks: [mnist]}  # `tasks` limits a scenario
        methods:
          fixed_lr: {}
          autotunenet: {n_startup_trials: 3}                            # constructor arguments
    """
    name: str
    epochs: int
    seeds: List[int]
    tasks: List[str]
    scenarios: Dict[str, dict]
    methods: Dict[str, dict]

    @classmethod
    def load(cls, path: Path) -> "Experiment":
        with open(path) as f:
            raw = yaml.safe_load(f)
        exp = cls(
            name=raw["name"],
            epochs=int(raw["epochs"]),
            seeds=[int(s) for s in raw["seeds"]],
            tasks=list(raw["tasks"]),
            scenarios={k: dict(v or {}) for k, v in raw["scenarios"].items()},
            methods={k: dict(v or {}) for k, v in raw["methods"].items()},
        )
        exp.validate()
        return exp

    def validate(self) -> None:
        for name in self.tasks:
            TASKS.get(name)
        for name, params in self.scenarios.items():
            SCENARIOS.get(name)
            for task in params.get("tasks", []):
                TASKS.get(task)
        for name in self.methods:
            METHODS.get(name)

    def scenario_params(self, name: str) -> dict:
        return {k: v for k, v in self.scenarios[name].items() if k != "tasks"}

    def build_method(self, name: str):
        return METHODS.get(name)(**self.methods[name])

    def cells(self, methods: Optional[Sequence[str]] = None, tasks: Optional[Sequence[str]] = None,
              scenarios: Optional[Sequence[str]] = None) -> List[Cell]:
        out = []
        for task in self.tasks:
            for scenario, params in self.scenarios.items():
                if "tasks" in params and task not in params["tasks"]:
                    continue
                for method in self.methods:
                    for seed in self.seeds:
                        cell = Cell(task, scenario, method, seed)
                        if methods and method not in methods:
                            continue
                        if tasks and task not in tasks:
                            continue
                        if scenarios and scenario not in scenarios:
                            continue
                        out.append(cell)
        return sorted(out)


def parse_shard(spec: str):
    """'2/4' -> (2, 4). Shards are 1-based."""
    try:
        index, count = (int(x) for x in spec.split("/"))
    except ValueError:
        raise ValueError(f"--shard must look like 2/4, got '{spec}'") from None
    if not 1 <= index <= count:
        raise ValueError(f"--shard index must be between 1 and {count}, got {index}")
    return index, count


def take_shard(cells: List[Cell], index: int, count: int) -> List[Cell]:
    """Round-robin split, so each shard gets a similar mix of tasks and methods."""
    return cells[index - 1::count]
