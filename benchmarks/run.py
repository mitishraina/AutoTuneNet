"""
Run a benchmark experiment.

    uv run python -m benchmarks.run --experiment benchmarks/experiments/comparison.yaml
    uv run python -m benchmarks.run --experiment ... --shard 2/4      # this machine runs a quarter
    uv run python -m benchmarks.run --experiment ... --list           # show the cells, run nothing

Each cell's result is written to
benchmarks/results/<experiment>/<method>/<task>/<scenario>/seed_<s>.json.
Finished cells are skipped, so an interrupted run resumes where it stopped.
"""
import argparse
import sys
import time
import traceback
from pathlib import Path

from benchmarks.core.device import resolve_device
from benchmarks.core.env import describe_environment
from benchmarks.core.experiment import Cell, Experiment, parse_shard, take_shard
from benchmarks.core.ledger import ComputeLedger
from benchmarks.core.method import RunContext
from benchmarks.core.registry import TASKS, load_plugins
from benchmarks.core.result import RunResult, result_path

DEFAULT_RESULTS = Path(__file__).resolve().parent / "results"


def run_cell(exp: Experiment, cell: Cell, device) -> RunResult:
    task = TASKS.get(cell.task)
    ctx = RunContext(
        task=task,
        scenario_name=cell.scenario,
        scenario_params=exp.scenario_params(cell.scenario),
        seed=cell.seed,
        device=device,
        epochs=exp.epochs,
    )
    method = exp.build_method(cell.method)
    ledger = ComputeLedger()

    start = time.perf_counter()
    outcome = method.run(ctx, ledger)
    ledger.wall_time_sec = time.perf_counter() - start

    return RunResult(
        experiment=exp.name,
        method=cell.method,
        task=cell.task,
        scenario=cell.scenario,
        seed=cell.seed,
        test_loss=outcome.test_loss,
        test_accuracy=outcome.test_accuracy,
        history=outcome.history,
        shift_epoch=ctx.new_scenario().shift_epoch,
        compute=ledger.to_dict(),
        environment=describe_environment(device),
        config={
            "epochs": exp.epochs,
            "method_params": exp.methods[cell.method],
            "scenario_params": exp.scenario_params(cell.scenario),
            "task": {"batch_size": task.batch_size, "lr": task.lr, "lr_range": list(task.lr_range)},
        },
        extra=outcome.extra,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run a benchmark experiment")
    parser.add_argument("--experiment", required=True, type=Path, help="experiment YAML file")
    parser.add_argument("--shard", default="1/1", help="run shard i of n, e.g. 2/4 (default 1/1)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--method", action="append", help="only this method (repeatable)")
    parser.add_argument("--task", action="append", help="only this task (repeatable)")
    parser.add_argument("--scenario", action="append", help="only this scenario (repeatable)")
    parser.add_argument("--force", action="store_true", help="re-run cells that already have results")
    parser.add_argument("--list", action="store_true", help="list this shard's cells and exit")
    args = parser.parse_args(argv)

    load_plugins()
    exp = Experiment.load(args.experiment)
    index, count = parse_shard(args.shard)
    cells = take_shard(exp.cells(args.method, args.task, args.scenario), index, count)

    def path_of(cell):
        return result_path(args.results_dir, exp.name, cell.method, cell.task, cell.scenario, cell.seed)

    if args.list:
        for cell in cells:
            print(f"{'done' if path_of(cell).exists() else 'todo'}  {cell.label()}")
        print(f"{len(cells)} cells in shard {index}/{count}")
        return 0

    device = resolve_device(args.device)
    print(f"[run] experiment={exp.name} shard={index}/{count} cells={len(cells)} device={device}")
    failed = []
    for i, cell in enumerate(cells, 1):
        path = path_of(cell)
        if path.exists() and not args.force:
            print(f"[{i}/{len(cells)}] skip (done): {cell.label()}")
            continue
        print(f"[{i}/{len(cells)}] run: {cell.label()}", flush=True)
        try:
            result = run_cell(exp, cell, device)
        except Exception:
            traceback.print_exc()
            failed.append(cell)
            continue
        result.save(path)
        print(f"    test_accuracy={result.test_accuracy:.4f} epochs={result.compute['epochs_trained']} "
              f"time={result.compute['wall_time_sec']:.0f}s", flush=True)

    if failed:
        print(f"[run] {len(failed)} cell(s) failed:")
        for cell in failed:
            print(f"    {cell.label()}")
        return 1
    print("[run] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
