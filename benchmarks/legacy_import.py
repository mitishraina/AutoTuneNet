"""
Convert the archived results of the previous benchmark scripts
(benchmarks/results/legacy/) into RunResult files, so the support matrix can
show them alongside the methods that still have to run.

    uv run python benchmarks/legacy_import.py          # -> benchmarks/results/previous/

Covered: fixed LR, cosine schedule and AutoTuneNet on MNIST and CIFAR-10, in
the stationary, label-noise, gradient-noise (MNIST) and bad-initial-LR
scenarios, 3 seeds, all trained on Apple MPS. The previous offline-HPO script
used a different protocol (8 x 3-epoch trials plus a 10-epoch final run), so
it is not mapped onto the new optuna_tpe method.
"""
import argparse
import json
from pathlib import Path

from autotunenet.bench.core.result import RunResult, result_path

RESULTS = Path(__file__).resolve().parent / "results"
LEGACY = RESULTS / "legacy"
EXPERIMENT = "previous"

METHOD = {"fixed_lr": "fixed_lr", "scheduler": "cosine", "autotunenet": "autotunenet"}
TASK_LR = {"mnist": 0.01, "cifar10": 0.001}
STRESS_LR = 0.1


def _load(path):
    with open(path) as f:
        return json.load(f)


def _history(train_loss, val_loss, val_accuracy, lrs):
    return [{"epoch": e, "lr": lrs[e], "train_loss": train_loss[e], "val_loss": val_loss[e],
             "val_accuracy": val_accuracy[e]} for e in range(len(val_loss))]


def _lrs_during(initial, after_each_epoch, epochs):
    """The old scripts logged the LR *after* each epoch's decision; shift to the LR used in it."""
    if not after_each_epoch:
        return [initial] * epochs
    return [initial] + list(after_each_epoch[: epochs - 1])


def _result(method, task, scenario, seed, test_loss, test_accuracy, history, shift_epoch, device, source, extra=None):
    return RunResult(
        experiment=EXPERIMENT, method=method, task=task, scenario=scenario, seed=seed,
        test_loss=test_loss, test_accuracy=test_accuracy, history=history, shift_epoch=shift_epoch,
        compute={"epochs_trained": len(history), "max_parallel_models": 1, "trials": 1, "restarts": 0,
                 "wall_time_sec": 0.0, "controller_time_sec": 0.0},
        environment={"device": device, "accelerator": "Apple MPS (previous benchmark scripts)",
                     "torch": None, "cuda": None, "python": None, "platform": None, "git_commit": None},
        config={"epochs": len(history), "source": source},
        extra={"imported_from": source, **(extra or {})},
    )


def stationary(out: Path):
    for legacy_method, method in METHOD.items():
        for run in sorted((LEGACY / legacy_method).glob("*/seed_*")):
            task, seed = run.parent.name, int(run.name.split("_")[1])
            m = _load(run / "metrics.json")
            lr_file = run / "lr_history.json"
            lrs = _lrs_during(TASK_LR[task], _load(lr_file) if lr_file.exists() else None, len(m["val_loss"]))
            yield _result(method, task, "stationary", seed, m["test_loss"], m["test_accuracy"],
                          _history(m["train_loss"], m["val_loss"], m["val_accuracy"], lrs), None,
                          _load(run / "run_info.json")["device"], str(run.relative_to(RESULTS)),
                          {"rollbacks": m["rollbacks"]} if "rollbacks" in m else None)


def stress(out: Path):
    for run in sorted((LEGACY / "stress_test").glob("*/seed_*")):
        task, seed = run.parent.name, int(run.name.split("_")[1])
        device = _load(run / "run_info.json")["device"]
        for legacy_method, r in _load(run / "results.json").items():
            lrs = _lrs_during(STRESS_LR, r.get("lr_history"), len(r["val_loss"]))
            yield _result(METHOD[legacy_method], task, "bad_initial_lr", seed, r["test_loss"], r["test_accuracy"],
                          _history(r["train_loss"], r["val_loss"], r["val_accuracy"], lrs), None, device,
                          str(run.relative_to(RESULTS)), {"rollbacks": r.get("rollbacks", 0)})


def shifted(out: Path):
    for config_dir in sorted(p for p in (LEGACY / "autotunenet_bench").iterdir() if p.is_dir()):
        name = config_dir.name
        task = "cifar10" if name.startswith("cifar") else "mnist"
        scenario = "gradient_noise" if "gradient_noise" in name else "label_noise"
        method = METHOD[name.rsplit("_", 1)[-1] if not name.endswith("fixed_lr") else "fixed_lr"]
        for run in sorted(config_dir.glob("seed_*")):
            seed = int(run.name.split("_")[1])
            rows = _load(run / "epoch_metric.json")
            test = _load(run / "test_metrics.json")
            lrs = _lrs_during(TASK_LR[task], [r["hyperparams"]["lr"] for r in rows], len(rows))
            yield _result(method, task, scenario, seed, test["test_loss"], test["test_accuracy"],
                          _history([r["train_loss"] for r in rows], [r["val_loss"] for r in rows],
                                   [r["val_accuracy"] for r in rows], lrs),
                          5, _load(run / "run_info.json")["device"], str(run.relative_to(RESULTS)))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Import previous benchmark results")
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    args = parser.parse_args(argv)
    count = 0
    for source in (stationary, stress, shifted):
        for r in source(args.results_dir):
            r.save(result_path(args.results_dir, EXPERIMENT, r.method, r.task, r.scenario, r.seed))
            count += 1
    print(f"imported {count} runs into {args.results_dir / EXPERIMENT}")


if __name__ == "__main__":
    main()
