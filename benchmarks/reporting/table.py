"""
The support matrix as data: one row per method, one column per task x
scenario, each cell holding the test-accuracy summary and its rank.
Renderers (image, Markdown, HTML) only format this structure.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from benchmarks.reporting.labels import (
    BASELINE, METHOD_ORDER, SCENARIO_ORDER, TASK_LABELS, SCENARIO_LABELS, label, ordered,
)


@dataclass
class Cell:
    mean: Optional[float] = None        # test accuracy in % (None: not run / not applicable)
    std: Optional[float] = None
    n: int = 0
    rank: Optional[int] = None          # 1 = best in its column
    worse_than_baseline: bool = False


@dataclass
class Matrix:
    columns: List[Tuple[str, str]]                      # (task, scenario)
    rows: List[str]                                     # methods, baseline last
    cells: Dict[Tuple[str, str, str], Cell]             # (method, task, scenario)
    cost: Dict[str, dict] = field(default_factory=dict)  # method -> epochs / parallel models / restarts
    tasks: List[str] = field(default_factory=list)

    def cell(self, method, task, scenario) -> Cell:
        return self.cells.get((method, task, scenario), Cell())

    def column_title(self, column) -> str:
        return label(SCENARIO_LABELS, column[1])

    def task_title(self, task) -> str:
        return label(TASK_LABELS, task)


def build_matrix(summary: dict) -> Matrix:
    groups = list(summary["groups"].values())
    tasks = ordered({g["task"] for g in groups}, list(TASK_LABELS))
    methods = ordered({g["method"] for g in groups}, METHOD_ORDER)
    columns = [(t, s) for t in tasks
               for s in ordered({g["scenario"] for g in groups if g["task"] == t}, SCENARIO_ORDER)]

    cells = {}
    for g in groups:
        acc = g["metrics"]["test_accuracy"]
        cells[(g["method"], g["task"], g["scenario"])] = Cell(
            mean=100 * acc["mean"], std=100 * acc["std"], n=acc["n"])

    for task, scenario in columns:
        present = [(m, cells[(m, task, scenario)]) for m in methods if (m, task, scenario) in cells]
        # Competition ranking on the displayed precision: equal scores share a rank (1, 2, 2, 4).
        shown = {m: round(c.mean, 2) for m, c in present}
        for m, c in present:
            c.rank = 1 + sum(v > shown[m] for v in shown.values())
        base = cells.get((BASELINE, task, scenario))
        if base is not None:
            for m, c in present:
                c.worse_than_baseline = m != BASELINE and c.mean < base.mean

    cost = {}
    for m in methods:
        rows = [g["metrics"] for g in groups if g["method"] == m]
        cost[m] = {
            "epochs": _mean([r["epochs_trained"]["mean"] for r in rows]),
            "parallel": max(r["max_parallel_models"]["mean"] for r in rows),
            "restarts": _mean([r["restarts"]["mean"] for r in rows]),
        }
    return Matrix(columns=columns, rows=methods, cells=cells, cost=cost, tasks=tasks)


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def cost_text(cost: dict) -> str:
    text = f"{cost['epochs']:.0f} epochs"
    if cost["parallel"] > 1:
        text += f" · {cost['parallel']:.0f} parallel"
    if cost["restarts"] >= 1:
        text += f" · {cost['restarts']:.0f} restarts"
    return text
