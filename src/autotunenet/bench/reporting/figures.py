"""Figures drawn from per-run histories: mean line with a +/- 1 std band over seeds."""
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from autotunenet.bench.reporting.labels import (  # noqa: E402
    METHOD_LABELS, METHOD_ORDER, SCENARIO_LABELS, TASK_LABELS, label, ordered,
)

# One colour per method everywhere; the paper's three methods keep the paper's colours.
METHOD_COLOURS = {
    "fixed_lr": "tab:blue", "cosine": "tab:orange", "autotunenet": "tab:green", "plateau": "tab:purple",
    "optuna_tpe": "tab:brown", "asha": "tab:pink", "hyperband": "tab:olive", "pbt": "tab:red",
}


def _by_cell(results):
    cells = defaultdict(list)
    for r in results:
        cells[(r.method, r.task, r.scenario)].append(r)
    return cells


def _finish(fig, ax, title, xlabel, ylabel, path: Path, note: str):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.text(0.99, 0.01, note, ha="right", va="bottom", fontsize=7, color="#555555")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def curves(results, task: str, scenario: str, key: str, path: Path, title: str,
           methods: Optional[Iterable[str]] = None, log_y: bool = False, ylabel: Optional[str] = None) -> bool:
    """Per-epoch `key` (val_loss, val_accuracy, lr) for each method; marks the shift epoch."""
    cells = _by_cell(results)
    available = {m for (m, t, s) in cells if t == task and s == scenario}
    methods = [m for m in (methods or ordered(available, METHOD_ORDER)) if m in available]
    if not methods:
        return False
    fig, ax = plt.subplots(figsize=(8, 5))
    shift, n = None, 0
    for m in methods:
        runs = cells[(m, task, scenario)]
        series = np.array([[h[key] for h in r.history] for r in runs], dtype=float)
        mean, std = series.mean(axis=0), series.std(axis=0)
        x = np.arange(len(mean))
        ax.plot(x, mean, color=METHOD_COLOURS.get(m), label=label(METHOD_LABELS, m), linewidth=1.8)
        ax.fill_between(x, mean - std, mean + std, color=METHOD_COLOURS.get(m), alpha=0.15, linewidth=0)
        shift, n = runs[0].shift_epoch, max(n, len(runs))
    if shift is not None:
        ax.axvline(shift - 0.5, color="#9a6700", linestyle="--", linewidth=1, label="shift")
    if log_y:
        ax.set_yscale("log")
    _finish(fig, ax, title, "Epoch", ylabel or key, path, f"mean ± std over {n} seeds")
    return True


def accuracy_vs_compute(results, task: str, scenario: str, path: Path) -> bool:
    """Test accuracy against total epochs trained: what each method buys for its compute."""
    cells = _by_cell(results)
    methods = ordered({m for (m, t, s) in cells if t == task and s == scenario}, METHOD_ORDER)
    if not methods:
        return False
    fig, ax = plt.subplots(figsize=(8, 5))
    n = 0
    for m in methods:
        runs = cells[(m, task, scenario)]
        acc = np.array([100 * r.test_accuracy for r in runs])
        epochs = np.mean([r.compute["epochs_trained"] for r in runs])
        ax.errorbar(epochs, acc.mean(), yerr=acc.std(), fmt="o", capsize=3, color=METHOD_COLOURS.get(m),
                    label=label(METHOD_LABELS, m))
        ax.annotate(label(METHOD_LABELS, m), (epochs, acc.mean()), textcoords="offset points", xytext=(6, 4),
                    fontsize=8)
        n = max(n, len(runs))
    ax.set_xscale("log")
    title = f"{label(TASK_LABELS, task)}, {label(SCENARIO_LABELS, scenario)}: accuracy vs compute"
    _finish(fig, ax, title, "Epochs trained per cell (all models, log scale)", "Test accuracy (%)", path,
            f"mean ± std over {n} seeds")
    return True


def paper_figures(results, task: str, out_dir: Path) -> List[Path]:
    """The four figures from the AutoTuneNet paper, for one task."""
    t = label(TASK_LABELS, task)
    trio = ["fixed_lr", "cosine", "autotunenet"]
    specs = [
        ("stationary", "val_loss", f"Validation Loss vs Epoch ({t})", "Validation Loss", trio, False, "fig1_val_loss"),
        ("stationary", "val_accuracy", f"Validation Accuracy vs Epoch ({t})", "Validation Accuracy", trio, False,
         "fig2_val_accuracy"),
        ("stationary", "lr", f"Learning Rate vs Epoch ({t})", "Learning Rate", ["cosine", "autotunenet"], False,
         "fig3_learning_rate"),
        ("bad_initial_lr", "val_loss", f"Stress Test: Validation Loss vs Epoch (Bad Initial LR, {t})",
         "Validation Loss", trio, False, "fig4_stress_test"),
    ]
    written = []
    for scenario, key, title, ylabel, methods, log_y, name in specs:
        path = out_dir / f"{task}_{name}.png"
        if curves(results, task, scenario, key, path, title, methods=methods, log_y=log_y, ylabel=ylabel):
            written.append(path)
    return written
