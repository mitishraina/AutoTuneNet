"""
Aggregate multi-seed benchmark results into mean ± std summaries.

Reads
    autotunenet_bench/results/<config>/seed_<s>/*.json
    benchmarks/<method>/results/<task>/seed_<s>/*.json
and writes
    autotunenet_bench/results/summary.json
    benchmarks/results_summary.json
plus Markdown tables on stdout.

Usage: python -m autotunenet_bench.aggregate
"""
import json
import statistics
from pathlib import Path

from autotunenet_bench.metrics.regret import RegretMetric

BENCH = Path("autotunenet_bench/results")
LEGACY = Path("benchmarks")
CONTROLLERS = ("autotunenet", "scheduler", "fixed_lr")
LEGACY_METHODS = ("fixed_lr", "scheduler", "offline_hpo", "autotunenet")


def _load(path: Path):
    with open(path) as f:
        return json.load(f)


def _stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
        "values": values,
    }


def _seed_dirs(root: Path):
    return sorted(d for d in root.glob("seed_*") if d.is_dir())


def _split(config: str):
    """'mnist_gradient_noise_autotunenet' -> ('mnist_gradient_noise', 'autotunenet')."""
    for c in CONTROLLERS:
        if config.endswith("_" + c):
            return config[: -len(c) - 1], c
    return config, None


def aggregate_bench():
    summary = {}
    for config_dir in sorted(p for p in BENCH.iterdir() if p.is_dir()):
        seeds = _seed_dirs(config_dir)
        if not seeds:
            continue
        setting, controller = _split(config_dir.name)
        rows = []
        for d in seeds:
            epochs = _load(d / "epoch_metric.json")
            stab = _load(d / "stability_metrics.json")
            comp = _load(d / "compute_metrics.json")
            test = _load(d / "test_metrics.json") if (d / "test_metrics.json").exists() else {}
            info = _load(d / "run_info.json")

            regret = None
            baseline = BENCH / f"{setting}_fixed_lr" / d.name / "epoch_metric.json"
            if controller != "fixed_lr" and baseline.exists():
                regret = RegretMetric(baseline).compute(d / "epoch_metric.json")["cumulative_regret"]

            rows.append({
                "seed": info["seed"],
                "device": info["device"],
                "final_val_accuracy": epochs[-1]["val_accuracy"],
                "best_val_accuracy": max(e["val_accuracy"] for e in epochs),
                "final_val_loss": epochs[-1]["val_loss"],
                "test_accuracy": test.get("test_accuracy"),
                "test_loss": test.get("test_loss"),
                "post_shift_degradation": stab.get("post_shift_degradation"),
                "loss_spike_magnitude": stab.get("loss_spike_magnitude"),
                "rollbacks": stab.get("rollback_count"),
                "controller_ms_per_epoch": comp.get("controller_time_per_epoch_ms"),
                "cumulative_regret_vs_fixed_lr": regret,
            })
        metrics = {k: _stats([r[k] for r in rows]) for k in rows[0] if k not in ("seed", "device")}
        summary[config_dir.name] = {
            "setting": setting,
            "controller": controller,
            "devices": sorted({r["device"] for r in rows}),
            "seeds": [r["seed"] for r in rows],
            "metrics": metrics,
            "per_seed": rows,
        }
    return summary


def aggregate_legacy():
    summary = {}
    for method in LEGACY_METHODS:
        for task_dir in sorted((LEGACY / method / "results").glob("*")):
            seeds = _seed_dirs(task_dir)
            if not seeds:
                continue
            rows = []
            for d in seeds:
                m = _load(d / "metrics.json")
                info = _load(d / "run_info.json")
                rows.append({
                    "seed": info["seed"],
                    "device": info["device"],
                    "test_accuracy": m["test_accuracy"],
                    "test_loss": m["test_loss"],
                    "final_val_accuracy": m["val_accuracy"][-1],
                    "best_val_accuracy": max(m["val_accuracy"]),
                    "epochs_trained": m["epochs_trained"],
                    "rollbacks": m.get("rollbacks"),
                })
            summary[f"{method}/{task_dir.name}"] = {
                "method": method,
                "task": task_dir.name,
                "devices": sorted({r["device"] for r in rows}),
                "seeds": [r["seed"] for r in rows],
                "metrics": {k: _stats([r[k] for r in rows]) for k in rows[0] if k not in ("seed", "device")},
                "per_seed": rows,
            }

    for task_dir in sorted((LEGACY / "stress_test" / "results").glob("*")):
        seeds = _seed_dirs(task_dir)
        if not seeds:
            continue
        per_method = {}
        for d in seeds:
            res = _load(d / "results.json")
            for name, r in res.items():
                per_method.setdefault(name, []).append({
                    "test_accuracy": r["test_accuracy"],
                    "test_loss": r["test_loss"],
                    "final_val_accuracy": r["val_accuracy"][-1],
                    "best_val_accuracy": max(r["val_accuracy"]),
                    "epochs_trained": r.get("epochs_trained"),
                    "rollbacks": r.get("rollbacks"),
                })
        for name, rows in per_method.items():
            summary[f"stress_test/{task_dir.name}/{name}"] = {
                "method": f"stress_test:{name}",
                "task": task_dir.name,
                "metrics": {k: _stats([r[k] for r in rows]) for k in rows[0]},
            }
    return summary


def _fmt(stat, scale=1.0, digits=2):
    if stat is None:
        return "—"
    return f"{stat['mean'] * scale:.{digits}f} ± {stat['std'] * scale:.{digits}f}"


def main():
    bench = aggregate_bench()
    legacy = aggregate_legacy()
    with open(BENCH / "summary.json", "w") as f:
        json.dump(bench, f, indent=2)
    with open(LEGACY / "results_summary.json", "w") as f:
        json.dump(legacy, f, indent=2)

    print("## AutoTuneNet-Bench (mean ± std over seeds)\n")
    print("| Config | n | Test acc | Final val acc | Final val loss | Post-shift degr. | Rollbacks | Regret vs fixed |")
    print("|---|---|---|---|---|---|---|---|")
    for name, s in bench.items():
        m = s["metrics"]
        print(f"| {name} | {m['final_val_accuracy']['n']} | {_fmt(m['test_accuracy'], 100)} | "
              f"{_fmt(m['final_val_accuracy'], 100)} | {_fmt(m['final_val_loss'], 1, 3)} | "
              f"{_fmt(m['post_shift_degradation'], 1, 3)} | {_fmt(m['rollbacks'], 1, 1)} | "
              f"{_fmt(m['cumulative_regret_vs_fixed_lr'], 1, 3)} |")

    print("\n## Legacy benchmarks (mean ± std over seeds)\n")
    print("| Method | Task | n | Test acc | Test loss | Best val acc | Epochs |")
    print("|---|---|---|---|---|---|---|")
    for name, s in legacy.items():
        m = s["metrics"]
        epochs = m.get("epochs_trained")
        print(f"| {s['method']} | {s['task']} | {m['test_accuracy']['n']} | {_fmt(m['test_accuracy'], 100)} | "
              f"{_fmt(m['test_loss'], 1, 3)} | {_fmt(m['best_val_accuracy'], 100)} | "
              f"{int(epochs['mean']) if epochs else 10} |")


if __name__ == "__main__":
    main()
