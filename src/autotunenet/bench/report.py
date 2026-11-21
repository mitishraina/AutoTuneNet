"""
Build an experiment's report from its results: summary.json, the support
matrix (support_matrix.svg / .md / .html) and figures, under
<results>/<experiment>/report/. Used by `autotunenet report`.
"""
import json
from pathlib import Path

from autotunenet.bench.aggregate import load_results, summarise
from autotunenet.bench.core.experiment import Experiment
from autotunenet.bench.reporting import figures, matrix as matrix_render
from autotunenet.bench.reporting.labels import SCENARIO_LABELS, TASK_LABELS, label
from autotunenet.bench.reporting.table import build_matrix


def plan_of(exp: Experiment) -> dict:
    """Planned (method, task, scenario) cells and seeds per cell, for progress in the matrix."""
    return {"cells": {(c.method, c.task, c.scenario) for c in exp.cells()}, "seeds": len(exp.seeds)}


def build_report(results_dir: Path, experiment: str, title: str, plan: dict | None = None) -> Path:
    results = load_results(results_dir, experiment)
    if not results and not plan:
        raise SystemExit(f"No results under {results_dir / experiment}")
    (results_dir / experiment).mkdir(parents=True, exist_ok=True)
    groups, warnings = summarise(results)
    summary = {"experiment": experiment, "warnings": warnings, "groups": groups}
    root = results_dir / experiment
    with open(root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    out = root / "report"
    for sub in ("accuracy_vs_compute", "curves", "paper_figures"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    m = build_matrix(summary, plan)
    seeds = plan["seeds"] if plan else max((g["metrics"]["test_accuracy"]["n"] for g in groups.values()), default=0)
    subtitle = f"{experiment} · {seeds} seed{'s' if seeds != 1 else ''} · test accuracy on held-out data"
    if plan:
        done, total = m.progress()
        subtitle += f" · {done} of {total} planned cells complete"
    svg = matrix_render.render_svg(m, title, subtitle=subtitle)
    (out / "support_matrix.svg").write_text(svg)
    (out / "support_matrix.md").write_text(matrix_render.render_markdown(m, title))
    (out / "support_matrix.html").write_text(matrix_render.render_html(svg, title))

    for task, scenario in m.columns:
        name = f"{task}_{scenario}"
        where = f"{label(TASK_LABELS, task)}, {label(SCENARIO_LABELS, scenario)}"
        figures.accuracy_vs_compute(results, task, scenario, out / "accuracy_vs_compute" / f"{name}.png")
        figures.curves(results, task, scenario, "val_loss", out / "curves" / f"{name}_val_loss.png",
                       f"{where}: validation loss", ylabel="Validation loss")
        figures.curves(results, task, scenario, "lr", out / "curves" / f"{name}_lr.png",
                       f"{where}: learning rate", log_y=True, ylabel="Learning rate (log)")
    for task in m.tasks:
        figures.paper_figures(results, task, out / "paper_figures")

    for w in warnings:
        print(f"WARNING: {w}")
    print(f"wrote {out}")
    return out
