"""
Plot multi-seed benchmark results (mean line, ±1 std band) for every task.

Usage: python -m benchmarks.final_results_graph.plot_results
Writes PNGs to benchmarks/final_results_graph/images/.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).parent / "images"
TASKS = ("mnist", "cifar10")
TASK_LABEL = {"mnist": "MNIST", "cifar10": "CIFAR-10"}
STYLE = {
    "fixed_lr": ("Fixed LR", "#6e7781"),
    "scheduler": ("Cosine scheduler", "#1f6feb"),
    "offline_hpo": ("Offline HPO (final run)", "#8250df"),
    "autotunenet": ("AutoTuneNet", "#d1242f"),
}


def _band(ax, series, key):
    if not series:
        return
    arr = np.array(series, dtype=float)
    mean, std = arr.mean(axis=0), arr.std(axis=0)
    x = np.arange(len(mean))
    label, color = STYLE[key]
    ax.plot(x, mean, color=color, label=f"{label} (n={len(series)})", linewidth=2)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)


def _seed_files(pattern_root: Path, name: str):
    return sorted(pattern_root.glob(f"seed_*/{name}"))


def _load(p):
    with open(p) as f:
        return json.load(f)


def _finish(ax, title, ylabel, path, shift_epoch=None):
    if shift_epoch is not None:
        ax.axvline(shift_epoch - 0.5, color="#9a6700", linestyle="--", linewidth=1, label="shift")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=150)
    plt.close(ax.figure)
    print("wrote", path)


def plot_legacy(task):
    _, ax = plt.subplots(figsize=(7, 4.2))
    for method in ("fixed_lr", "scheduler", "offline_hpo", "autotunenet"):
        files = _seed_files(Path("benchmarks") / method / "results" / task, "metrics.json")
        _band(ax, [_load(f)["val_accuracy"] for f in files], method)
    _finish(ax, f"{TASK_LABEL[task]}: validation accuracy (stationary)", "Validation accuracy",
            OUT / f"{task}_legacy_val_accuracy.png")

    _, ax = plt.subplots(figsize=(7, 4.2))
    for method in ("scheduler", "autotunenet"):
        files = _seed_files(Path("benchmarks") / method / "results" / task, "lr_history.json")
        _band(ax, [_load(f) for f in files], method)
    ax.set_yscale("log")
    _finish(ax, f"{TASK_LABEL[task]}: learning rate (stationary)", "Learning rate (log)",
            OUT / f"{task}_legacy_lr.png")


def plot_bench(task, setting, title):
    _, ax = plt.subplots(figsize=(7, 4.2))
    found = False
    for controller in ("fixed_lr", "scheduler", "autotunenet"):
        root = Path("autotunenet_bench/results") / f"{setting}_{controller}"
        files = _seed_files(root, "epoch_metric.json")
        if files:
            found = True
        _band(ax, [[e["val_loss"] for e in _load(f)] for f in files], controller)
    if found:
        _finish(ax, title, "Validation loss", OUT / f"{setting}_val_loss.png", shift_epoch=5)
    else:
        plt.close(ax.figure)


# ---------------------------------------------------------------- paper figures
# Same four figures as the AutoTuneNet paper (Figs. 1-4), regenerated from the
# multi-seed runs: mean line and +/- 1 std band. Colours follow the paper.
PAPER_STYLE = {
    "fixed_lr": ("Fixed LR", "tab:blue"),
    "scheduler": ("Scheduler", "tab:orange"),
    "autotunenet": ("AutoTuneNet", "tab:green"),
}


def _paper_band(ax, series, key):
    if not series:
        return
    arr = np.array(series, dtype=float)
    mean, std = arr.mean(axis=0), arr.std(axis=0)
    x = np.arange(len(mean))
    label, color = PAPER_STYLE[key]
    ax.plot(x, mean, color=color, label=label, linewidth=1.8)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)


def _paper_finish(fig, ax, title, ylabel, path, n):
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()
    fig.text(0.99, 0.01, f"mean ± std over {n} seeds", ha="right", va="bottom", fontsize=8, color="#555555")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)


def plot_paper_figures(task):
    label = TASK_LABEL[task]
    out = OUT / "paper_figures"
    out.mkdir(parents=True, exist_ok=True)

    def legacy(method, name):
        return [_load(f) for f in _seed_files(Path("benchmarks") / method / "results" / task, name)]

    runs = {m: legacy(m, "metrics.json") for m in ("fixed_lr", "scheduler", "autotunenet")}
    n = min(len(v) for v in runs.values())
    if n == 0:
        return

    for key, ylabel, fig_no, fname in [
        ("val_loss", "Validation Loss", 1, "fig1_val_loss"),
        ("val_accuracy", "Validation Accuracy", 2, "fig2_val_accuracy"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for method, metrics in runs.items():
            _paper_band(ax, [m[key] for m in metrics], method)
        _paper_finish(fig, ax, f"{ylabel} vs Epoch ({label})", ylabel, out / f"{task}_{fname}.png", n)

    fig, ax = plt.subplots(figsize=(8, 5))
    for method in ("scheduler", "autotunenet"):
        _paper_band(ax, legacy(method, "lr_history.json"), method)
    _paper_finish(fig, ax, f"Learning Rate vs Epoch ({label})", "Learning Rate",
                  out / f"{task}_fig3_learning_rate.png", n)

    stress = legacy("stress_test", "results.json")
    if stress:
        fig, ax = plt.subplots(figsize=(8, 5))
        for method in ("fixed_lr", "scheduler", "autotunenet"):
            _paper_band(ax, [r[method]["val_loss"] for r in stress], method)
        _paper_finish(fig, ax, f"Stress Test: Validation Loss vs Epoch (Bad Initial LR, {label})",
                      "Validation Loss", out / f"{task}_fig4_stress_test.png", len(stress))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        plot_paper_figures(task)
    for task in TASKS:
        plot_legacy(task)
    plot_bench("mnist", "mnist", "MNIST: validation loss, 30% label noise from epoch 5")
    plot_bench("mnist", "mnist_gradient_noise", "MNIST: validation loss, gradient noise from epoch 5")
    plot_bench("cifar10", "cifar", "CIFAR-10: validation loss, 30% label noise from epoch 5")


if __name__ == "__main__":
    main()
