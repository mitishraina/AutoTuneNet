"""
Build an experiment's report: summary, support matrix and figures.

    uv run python -m benchmarks.report --experiment comparison

Writes to benchmarks/results/<experiment>/report/:
    support_matrix.svg   (primary; colour-coded by rank, see benchmarks/reporting/matrix.py)
    support_matrix.md    (plain-text table)
    support_matrix.html  (page embedding the SVG)
    accuracy_vs_compute/<task>_<scenario>.png
    curves/<task>_<scenario>_{val_loss,lr}.png
    paper_figures/<task>_fig{1..4}_*.png
"""
import argparse
import json
from pathlib import Path

from benchmarks.aggregate import DEFAULT_RESULTS, load_results, summarise
from benchmarks.reporting import figures, matrix as matrix_render
from benchmarks.reporting.labels import SCENARIO_LABELS, TASK_LABELS, label
from benchmarks.reporting.table import build_matrix


def build_report(results_dir: Path, experiment: str, title: str) -> Path:
    results = load_results(results_dir, experiment)
    if not results:
        raise SystemExit(f"No results under {results_dir / experiment}")
    groups, warnings = summarise(results)
    summary = {"experiment": experiment, "warnings": warnings, "groups": groups}
    root = results_dir / experiment
    with open(root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    out = root / "report"
    for sub in ("accuracy_vs_compute", "curves", "paper_figures"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    m = build_matrix(summary)
    seeds = max(g["metrics"]["test_accuracy"]["n"] for g in groups.values())
    seed_text = f"{seeds} seed" + ("s" if seeds != 1 else "")
    svg = matrix_render.render_svg(m, title, subtitle=f"{experiment} · {seed_text} · test accuracy on held-out data")
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


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the support matrix and figures for an experiment")
    parser.add_argument("--experiment", default="comparison")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--title", default="Support matrix")
    args = parser.parse_args(argv)
    build_report(args.results_dir, args.experiment, args.title)


if __name__ == "__main__":
    main()
