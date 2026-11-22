"""
Run the cells of an experiment on this machine.

Each cell's result is written to <results>/<experiment>/<method>/<task>/<scenario>/seed_<s>.json.
Cells that already have a result are skipped, so an interrupted run resumes
where it stopped. Used by `autotunenet run` and by the hosted runtimes.
"""
import json
import time
import traceback
from pathlib import Path
from typing import List, Optional, Sequence

from autotunenet.bench import progress
from autotunenet.bench.core.device import resolve_device
from autotunenet.bench.core.env import describe_environment
from autotunenet.bench.core.experiment import Cell, Experiment
from autotunenet.bench.core.ledger import ComputeLedger
from autotunenet.bench.core.method import RunContext
from autotunenet.bench.core.registry import TASKS
from autotunenet.bench.core.result import RunResult, result_path


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


def path_of(results_dir: Path, exp: Experiment, cell: Cell) -> Path:
    return result_path(results_dir, exp.name, cell.method, cell.task, cell.scenario, cell.seed)


def pending_cells(exp: Experiment, results_dir: Path, cells: Optional[Sequence[Cell]] = None) -> List[Cell]:
    """Cells without a result yet (all of the experiment's cells by default)."""
    cells = exp.cells() if cells is None else cells
    return [c for c in cells if not path_of(results_dir, exp, c).exists()]


def read_cells(path: Path) -> List[Cell]:
    return [Cell(**c) for c in json.loads(Path(path).read_text())]


def execute(exp: Experiment, cells: Sequence[Cell], device_name: str, results_dir: Path,
            force: bool = False) -> List[Cell]:
    """Run the given cells (skipping finished ones unless force); returns the cells that failed."""
    todo = cells if force else pending_cells(exp, results_dir, cells)
    device = resolve_device(device_name)
    progress.run_header(exp.name, len(todo), len(cells), device)
    failed = []
    for i, cell in enumerate(todo, 1):
        progress.run_start(i, len(todo), cell)
        progress.emit("start", i=i, n=len(todo), **cell.to_dict())
        start = time.perf_counter()
        try:
            result = run_cell(exp, cell, device)
        except Exception as e:
            traceback.print_exc()
            error = f"{type(e).__name__}: {e}".splitlines()[0][:200]
            progress.run_failed(i, len(todo), cell, error)
            progress.emit("failed", i=i, n=len(todo), error=error, **cell.to_dict())
            failed.append(cell)
            continue
        result.save(path_of(results_dir, exp, cell))
        seconds = time.perf_counter() - start
        progress.run_done(i, len(todo), cell, result.test_accuracy, result.compute["epochs_trained"], seconds)
        progress.emit("done", i=i, n=len(todo), accuracy=result.test_accuracy,
                      epochs=result.compute["epochs_trained"], seconds=seconds, **cell.to_dict())
    if failed:
        print(f"{len(failed)} cell(s) failed: " + "; ".join(c.label() for c in failed), flush=True)
    return failed
