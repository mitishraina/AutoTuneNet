"""
The `autotunenet` command.

    autotunenet --help              every command, one line each
    autotunenet <command> --help    what a command does, its options and examples
"""
import argparse
import sys
from pathlib import Path

TOP = """\
AutoTuneNet benchmarks: compare learning-rate strategies (AutoTuneNet, fixed LR,
schedules, Optuna, ASHA, Hyperband, PBT) on your tasks, on this machine or on Kaggle.

Most people only need:
  autotunenet run --experiment comparison                     train here
  autotunenet run --experiment comparison --runtime kaggle    train on Kaggle (your account)

Run 'autotunenet <command> --help' for details on a command."""

COMMANDS = {
    "list": (
        "Show the built-in experiments, tasks, scenarios, methods and runtimes",
        """\
Lists what you can use in an experiment file or on the command line: the
built-in experiments (usable by name with --experiment), tasks (dataset +
model), scenarios (what happens during training), methods (how the learning
rate is chosen) and runtimes (where training happens). Tasks, scenarios and
methods you add with --plugins are listed too.""",
        ["autotunenet list", "autotunenet list --plugins my_tasks.py"],
    ),
    "setup": (
        "Check that a runtime is ready to use",
        """\
Checks what the chosen runtime needs and says exactly what is missing.
On this machine it reports which device training will use (CUDA GPU, Apple GPU
or CPU). For Kaggle it checks the Kaggle CLI and that a token is configured
under ~/.kaggle/ (the token itself is never read or printed).""",
        ["autotunenet setup", "autotunenet setup --runtime kaggle"],
    ),
    "data": (
        "Fetch the datasets an experiment needs, and upload them for a hosted runtime",
        """\
Downloads each task's data on this machine if it is missing (into ./data, or
$AUTOTUNENET_DATA_DIR). With --runtime kaggle it also uploads each task's data
once, as a private dataset autotunenet-<task> on your account; tasks already
uploaded are skipped unless --force. Choose the tasks with --task, or take
those of an experiment with --experiment.""",
        ["autotunenet data --experiment comparison",
         "autotunenet data --task cifar10 --runtime kaggle",
         "autotunenet data --task my_task --plugins my_tasks.py --runtime kaggle --force"],
    ),
    "prepare": (
        "Plan the unfinished cells of an experiment for a runtime",
        """\
Works out which cells (method x task x scenario x seed) have no result yet and
plans only those. For Kaggle it writes the notebooks (two GPUs each, --parallel
notebooks) to <results>/<experiment>/notebooks/ so you can read exactly what
will run, without starting anything. If a previous round is still running, it
is kept and continued instead.""",
        ["autotunenet prepare --experiment comparison --runtime kaggle --parallel 3"],
    ),
    "start": (
        "Start what prepare planned",
        """\
Pushes the prepared notebooks to Kaggle and starts them; notebooks already
started are skipped. On this machine there is nothing to start: use run.""",
        ["autotunenet start --experiment comparison --runtime kaggle"],
    ),
    "status": (
        "Show how far an experiment is: cells done and what is running",
        """\
Counts the cells with results and lists the cells still missing per method.
For Kaggle it also shows each notebook's state (queued, running, complete,
error, downloaded).""",
        ["autotunenet status --experiment comparison", "autotunenet status --experiment comparison --runtime kaggle"],
    ),
    "pull": (
        "Download the results of finished notebooks",
        """\
Fetches the results of every finished Kaggle notebook into your results folder
(existing results are kept) and keeps the notebooks' logs locally.""",
        ["autotunenet pull --experiment comparison --runtime kaggle"],
    ),
    "report": (
        "Build the support matrix and figures from the results so far",
        """\
Summarises the results (mean +/- std over seeds) and writes, under
<results>/<experiment>/report/: support_matrix.svg (with every planned cell,
so unfinished ones show as pending), support_matrix.md, support_matrix.html,
accuracy-vs-compute plots, learning-rate and validation-loss curves, and the
paper's figures. Prints the matrix as a table.

--plan draws the progress of one results folder against another experiment's
plan, e.g. the imported earlier runs against the full comparison.""",
        ["autotunenet report --experiment comparison",
         "autotunenet report --experiment previous --plan comparison"],
    ),
    "run": (
        "Do everything: check, fetch data, plan, train, collect results, report",
        """\
Runs every stage for the cells that have no result yet. Without --runtime it
trains here (CUDA GPU, Apple GPU or CPU). With --runtime kaggle it checks your
account, uploads missing data, prepares notebooks, starts them, follows their
progress, downloads results as notebooks finish and builds the report.

Everything is resumable: stop it at any point and run the same command again
to continue where it stopped. Finished cells are never run twice.""",
        ["autotunenet run --experiment smoke                       (seconds, toy task)",
         "autotunenet run --experiment comparison",
         "autotunenet run --experiment comparison --method pbt --task cifar10",
         "autotunenet run --experiment comparison --runtime kaggle --parallel 3",
         "autotunenet run --experiment my_experiment.yaml --plugins my_tasks.py"],
    ),
}


def _add_common(p, experiment=True, runtime=True, filters=False, required=True):
    if experiment:
        p.add_argument("-e", "--experiment", required=required, metavar="NAME_OR_FILE",
                       help="built-in experiment name (see `autotunenet list`) or a YAML file")
    if runtime:
        p.add_argument("--runtime", default="machine", metavar="NAME",
                       help="where to train: kaggle; omit to use this machine")
    if filters:
        p.add_argument("--method", action="append", metavar="NAME", help="only this method (repeatable)")
        p.add_argument("--task", action="append", metavar="NAME", help="only this task (repeatable)")
        p.add_argument("--scenario", action="append", metavar="NAME", help="only this scenario (repeatable)")
    p.add_argument("--plugins", action="append", default=[], metavar="FILE",
                   help="Python file or module adding your own tasks, scenarios or methods (repeatable)")
    p.add_argument("--results-dir", type=Path, metavar="DIR",
                   help="where results go (default: benchmarks/results in this repository, "
                        "./autotunenet_results elsewhere, or $AUTOTUNENET_RESULTS)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autotunenet", description=TOP,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", metavar="<command>", title="commands")
    for name, (summary, details, examples) in COMMANDS.items():
        p = sub.add_parser(name, help=summary, description=f"{summary}.\n\n{details}",
                           epilog="examples:\n" + "\n".join(f"  {e}" for e in examples),
                           formatter_class=argparse.RawDescriptionHelpFormatter)
        if name == "list":
            _add_common(p, experiment=False, runtime=False)
        elif name == "setup":
            _add_common(p, experiment=False)
        elif name == "data":
            _add_common(p, required=False)
            p.add_argument("--task", action="append", metavar="NAME", help="task to fetch (repeatable)")
            p.add_argument("--force", action="store_true", help="upload again even if already uploaded")
        elif name in ("prepare", "run"):
            _add_common(p, filters=True)
            p.add_argument("--parallel", type=int, default=3, metavar="N",
                           help="Kaggle notebooks to run at once, each on 2 GPUs (default 3)")
            if name == "run":
                p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
                               help="device on this machine (default: best available)")
                p.add_argument("--no-report", action="store_true", help="skip building the report at the end")
                p.add_argument("--cells", type=Path, help=argparse.SUPPRESS)
        elif name == "report":
            _add_common(p, experiment=False, runtime=False)
            p.add_argument("-e", "--experiment", required=True, metavar="NAME_OR_FILE",
                           help="experiment (or results folder name) to report on")
            p.add_argument("--plan", metavar="NAME_OR_FILE",
                           help="experiment whose cells count as planned (default: --experiment)")
            p.add_argument("--title", default="Support matrix", help="title of the support matrix")
        else:
            _add_common(p, filters=(name == "status"))
    return parser


# ------------------------------------------------------------------ commands

def _context(args, load_experiment=True):
    from autotunenet.bench import paths
    from autotunenet.bench.core.experiment import Experiment, resolve_experiment
    from autotunenet.bench.core.registry import load_plugins
    from autotunenet.bench.plugins import load_user_plugins

    load_plugins()
    load_user_plugins(args.plugins)
    results_dir = args.results_dir or paths.results_dir()
    exp = exp_path = None
    if load_experiment and getattr(args, "experiment", None):
        exp_path = resolve_experiment(args.experiment)
        exp = Experiment.load(exp_path)
    return exp, exp_path, results_dir


def _cells(args, exp):
    if getattr(args, "cells", None):
        from autotunenet.bench.run import read_cells
        return read_cells(args.cells)
    return exp.cells(getattr(args, "method", None), getattr(args, "task", None), getattr(args, "scenario", None))


def _runtime(args):
    from autotunenet.bench.runtimes import get_runtime
    return get_runtime(getattr(args, "runtime", "machine"))


def cmd_list(args):
    from autotunenet.bench.core.experiment import BUILT_IN
    from autotunenet.bench.core.registry import METHODS, SCENARIOS, TASKS
    from autotunenet.bench.runtimes import RUNTIMES
    _context(args)
    rows = [("experiments", sorted(p.stem for p in BUILT_IN.glob("*.yaml"))), ("tasks", TASKS.names()),
            ("scenarios", SCENARIOS.names()), ("methods", METHODS.names()), ("runtimes", list(RUNTIMES))]
    for title, names in rows:
        print(f"{title:12s} {', '.join(names)}")


def cmd_setup(args):
    _context(args)
    problems = _runtime(args).setup()
    for p in problems:
        print(f"[setup] {p}")
    print("[setup] ready" if not problems else "[setup] not ready")
    return 1 if problems else 0


def cmd_data(args):
    from autotunenet.bench.core.registry import TASKS
    exp, _, _ = _context(args)
    names = args.task or (sorted({c.task for c in exp.cells()}) if exp else [])
    if not names:
        raise SystemExit("Choose tasks with --task, or an experiment with --experiment.")
    tasks = [TASKS.get(n) for n in names]
    from autotunenet.bench.runtimes.machine import Machine
    Machine().data(tasks)
    if args.runtime != "machine":
        _runtime(args).data(tasks, force=args.force)


def cmd_prepare(args):
    exp, exp_path, results_dir = _context(args)
    _runtime(args).prepare(exp, exp_path, _cells(args, exp), results_dir, args.plugins, args.parallel)


def cmd_start(args):
    exp, _, results_dir = _context(args)
    _runtime(args).start(exp, results_dir)


def cmd_pull(args):
    exp, _, results_dir = _context(args)
    print(f"[pull] {_runtime(args).pull(exp, results_dir)} new result(s)")


def cmd_status(args):
    from collections import Counter
    from autotunenet.bench.run import pending_cells
    exp, _, results_dir = _context(args)
    cells = _cells(args, exp)
    missing = pending_cells(exp, results_dir, cells)
    print(f"{exp.name}: {len(cells) - len(missing)}/{len(cells)} cells done")
    for method, n in sorted(Counter(c.method for c in missing).items()):
        print(f"  {method}: {n} to run")
    for line in _runtime(args).describe(exp, results_dir):
        print(line)


def cmd_report(args):
    from autotunenet.bench.core.experiment import BUILT_IN, Experiment
    from autotunenet.bench.report import build_report, plan_of
    _, _, results_dir = _context(args, load_experiment=False)
    spec = args.experiment
    is_experiment = Path(spec).is_file() or (BUILT_IN / f"{spec}.yaml").is_file()
    name = Experiment.load(spec).name if is_experiment else spec
    plan_spec = args.plan or (spec if is_experiment else None)
    plan = plan_of(Experiment.load(plan_spec)) if plan_spec else None
    out = build_report(results_dir, name, args.title, plan)
    print((out / "support_matrix.md").read_text())


def cmd_run(args):
    exp, exp_path, results_dir = _context(args)
    cells = _cells(args, exp)
    left = _runtime(args).run(exp, exp_path, cells, results_dir, args.plugins, args.parallel, args.device)
    if not args.no_report:
        from autotunenet.bench.report import build_report, plan_of
        if any((results_dir / exp.name).glob("*/*/*/seed_*.json")):
            out = build_report(results_dir, exp.name, "Support matrix", plan_of(exp))
            print((out / "support_matrix.md").read_text())
    if left:
        print(f"{len(left)} cell(s) are not finished; run the same command again to continue.")
        return 1
    print(f"{exp.name}: all {len(cells)} cells done")
    return 0


HANDLERS = {"list": cmd_list, "setup": cmd_setup, "data": cmd_data, "prepare": cmd_prepare, "start": cmd_start,
            "status": cmd_status, "pull": cmd_pull, "report": cmd_report, "run": cmd_run}


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    try:
        return HANDLERS[args.command](args) or 0
    except KeyboardInterrupt:
        print("\nStopped. Run the same command again to continue where it left off.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
