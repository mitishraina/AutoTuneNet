"""
Summarise an experiment's results as mean ± std over seeds.

    uv run python -m benchmarks.aggregate --experiment comparison

Reads only RunResult files under benchmarks/results/<experiment>/ and writes
benchmarks/results/<experiment>/summary.json, plus a Markdown table on stdout.
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from benchmarks.core.metrics import cumulative_regret, post_shift_metrics, val_losses
from benchmarks.core.result import RunResult

DEFAULT_RESULTS = Path(__file__).resolve().parent / "results"
BASELINE = "fixed_lr"


def load_results(root: Path, experiment: str):
    return [RunResult.load(p) for p in sorted((root / experiment).glob("*/*/*/seed_*.json"))]


def stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "n": len(values), "values": values}


def summarise(results):
    baselines = {(r.task, r.scenario, r.seed): r for r in results if r.method == BASELINE}
    groups = defaultdict(list)
    for r in results:
        groups[(r.method, r.task, r.scenario)].append(r)

    summary, warnings = {}, []
    accelerators = defaultdict(set)
    for (method, task, scenario), runs in sorted(groups.items()):
        per_seed = []
        for r in runs:
            accelerators[(task, scenario)].add(r.environment.get("accelerator"))
            base = baselines.get((r.task, r.scenario, r.seed))
            regret = None
            if base is not None and method != BASELINE:
                regret = cumulative_regret(val_losses(r.history), val_losses(base.history))
            shift = post_shift_metrics(val_losses(r.history), r.shift_epoch)
            per_seed.append({
                "seed": r.seed,
                "test_accuracy": r.test_accuracy,
                "test_loss": r.test_loss,
                "final_val_accuracy": r.history[-1]["val_accuracy"],
                "epochs_trained": r.compute["epochs_trained"],
                "max_parallel_models": r.compute["max_parallel_models"],
                "trials": r.compute["trials"],
                "wall_time_sec": r.compute["wall_time_sec"],
                "regret_vs_fixed_lr": regret,
                "post_shift_degradation": shift["degradation"],
                "recovery_epochs": shift["recovery_epochs"],
            })
        metrics = {k: stats([row[k] for row in per_seed]) for k in per_seed[0] if k != "seed"}
        summary[f"{method}/{task}/{scenario}"] = {
            "method": method, "task": task, "scenario": scenario,
            "metrics": metrics, "per_seed": per_seed,
        }
    for (task, scenario), accs in accelerators.items():
        if len(accs) > 1:
            warnings.append(f"{task}/{scenario} mixes hardware: {sorted(map(str, accs))}")
    return summary, warnings


def fmt(stat, scale=1.0, digits=2):
    if stat is None:
        return "—"
    return f"{stat['mean'] * scale:.{digits}f} ± {stat['std'] * scale:.{digits}f}"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Summarise benchmark results")
    parser.add_argument("--experiment", default="comparison")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args(argv)

    results = load_results(args.results_dir, args.experiment)
    if not results:
        raise SystemExit(f"No results under {args.results_dir / args.experiment}")
    summary, warnings = summarise(results)
    out = args.results_dir / args.experiment / "summary.json"
    with open(out, "w") as f:
        json.dump({"experiment": args.experiment, "warnings": warnings, "groups": summary}, f, indent=2)

    print("| Method | Task | Scenario | n | Test acc | Epochs | Regret vs fixed LR |")
    print("|---|---|---|---|---|---|---|")
    for s in summary.values():
        m = s["metrics"]
        print(f"| {s['method']} | {s['task']} | {s['scenario']} | {m['test_accuracy']['n']} | "
              f"{fmt(m['test_accuracy'], 100)} | {fmt(m['epochs_trained'], 1, 0)} | "
              f"{fmt(m['regret_vs_fixed_lr'], 1, 3)} |")
    for w in warnings:
        print(f"WARNING: {w}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
