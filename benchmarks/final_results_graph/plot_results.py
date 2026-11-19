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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        plot_legacy(task)
    plot_bench("mnist", "mnist", "MNIST: validation loss, 30% label noise from epoch 5")
    plot_bench("mnist", "mnist_gradient_noise", "MNIST: validation loss, gradient noise from epoch 5")
    plot_bench("cifar10", "cifar", "CIFAR-10: validation loss, 30% label noise from epoch 5")


if __name__ == "__main__":
    main()
