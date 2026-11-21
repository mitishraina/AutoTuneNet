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


DONE, PARTIAL, PENDING, NA = "done", "partial", "pending", "n/a"


@dataclass
class Cell:
    status: str = NA                    # done | partial (some seeds) | pending (planned, not run) | n/a
    mean: Optional[float] = None        # test accuracy in %
    std: Optional[float] = None
    n: int = 0                          # seeds with results
    planned: int = 0                    # seeds planned (0 when there is no plan)
    rank: Optional[int] = None          # 1 = best in its column (among cells with results)
    delta: Optional[float] = None       # percentage points vs the baseline in the same column
    worse_than_baseline: bool = False


@dataclass
class Matrix:
    columns: List[Tuple[str, str]]                      # (task, scenario)
    rows: List[str]                                     # methods, baseline last
    cells: Dict[Tuple[str, str, str], Cell]             # (method, task, scenario)
    cost: Dict[str, Optional[dict]] = field(default_factory=dict)  # method -> epochs / parallel / restarts
    tasks: List[str] = field(default_factory=list)

    def progress(self) -> Tuple[int, int]:
        """(cells complete, cells planned); without a plan every cell with results counts."""
        planned = [c for c in self.cells.values() if c.status != NA]
        return sum(c.status == DONE for c in planned), len(planned)

    def cell(self, method, task, scenario) -> Cell:
        return self.cells.get((method, task, scenario), Cell())

    def column_title(self, column) -> str:
        return label(SCENARIO_LABELS, column[1])

    def task_title(self, task) -> str:
        return label(TASK_LABELS, task)


def build_matrix(summary: dict, plan: Optional[dict] = None) -> Matrix:
    """
    plan (optional): {"cells": {(method, task, scenario), ...}, "seeds": int} from
    an experiment file. With a plan, every planned cell appears: with results,
    as partial (fewer seeds than planned) or as pending (not run yet).
    """
    groups = list(summary["groups"].values())
    planned = set(plan["cells"]) if plan else set()
    planned_seeds = plan["seeds"] if plan else 0
    keys = {(g["method"], g["task"], g["scenario"]) for g in groups} | planned
    tasks = ordered({t for _, t, _ in keys}, list(TASK_LABELS))
    methods = ordered({m for m, _, _ in keys}, METHOD_ORDER)
    columns = [(t, s) for t in tasks for s in ordered({s for _, tt, s in keys if tt == t}, SCENARIO_ORDER)]

    cells = {key: Cell(status=PENDING, planned=planned_seeds) for key in planned}
    for g in groups:
        acc = g["metrics"]["test_accuracy"]
        key = (g["method"], g["task"], g["scenario"])
        complete = not plan or key not in planned or acc["n"] >= planned_seeds
        cells[key] = Cell(status=DONE if complete else PARTIAL, mean=100 * acc["mean"], std=100 * acc["std"],
                          n=acc["n"], planned=planned_seeds if key in planned else 0)

    for task, scenario in columns:
        present = [(m, cells[(m, task, scenario)]) for m in methods
                   if (m, task, scenario) in cells and cells[(m, task, scenario)].mean is not None]
        # Competition ranking on the displayed precision: equal scores share a rank (1, 2, 2, 4).
        shown = {m: round(c.mean, 2) for m, c in present}
        for m, c in present:
            c.rank = 1 + sum(v > shown[m] for v in shown.values())
        base = cells.get((BASELINE, task, scenario))
        if base is not None and base.mean is not None:
            for m, c in present:
                if m != BASELINE:
                    c.delta = c.mean - base.mean
                    c.worse_than_baseline = c.delta < 0

    cost = {}
    for m in methods:
        rows = [g["metrics"] for g in groups if g["method"] == m]
        if not rows:
            cost[m] = None
            continue
        cost[m] = {
            "epochs": _mean([r["epochs_trained"]["mean"] for r in rows]),
            "parallel": max(r["max_parallel_models"]["mean"] for r in rows),
            "restarts": _mean([r["restarts"]["mean"] for r in rows]),
        }
    return Matrix(columns=columns, rows=methods, cells=cells, cost=cost, tasks=tasks)


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def cost_text(cost: Optional[dict]) -> str:
    if cost is None:
        return "—"
    text = f"{cost['epochs']:.0f} epochs"
    if cost["parallel"] > 1:
        text += f" · {cost['parallel']:.0f} parallel"
    if cost["restarts"] >= 1:
        text += f" · {cost['restarts']:.0f} restarts"
    return text
