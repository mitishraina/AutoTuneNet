"""Summarise an experiment's results: mean ± std over seeds per method, task and scenario."""
import statistics
from collections import defaultdict
from pathlib import Path

from autotunenet.bench.core.metrics import cumulative_regret, post_shift_metrics, val_losses
from autotunenet.bench.core.result import RunResult

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
                "restarts": r.compute["restarts"],
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
