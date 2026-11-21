# Command line

The `autotunenet` command is installed with the package. `autotunenet --help` lists the commands. `autotunenet <command> --help` shows one command's description, options and examples.

```text
$ autotunenet --help
usage: autotunenet [-h] <command> ...

commands:
    list      Show the built-in experiments, tasks, scenarios, methods and runtimes
    setup     Check that a runtime is ready to use
    data      Fetch the datasets an experiment needs, and upload them for a hosted runtime
    prepare   Plan the unfinished cells of an experiment for a runtime
    start     Start what prepare planned
    status    Show how far an experiment is: cells done and what is running
    pull      Download the results of finished notebooks
    report    Build the support matrix and figures from the results so far
    run       Do everything: check, fetch data, plan, train, collect results, report
```

In a source checkout, prefix commands with `uv run`. `python -m autotunenet` is equivalent to `autotunenet`.

## Common options

| Option | Commands | Meaning |
|---|---|---|
| `-e, --experiment NAME_OR_FILE` | all but `list`, `setup` | A built-in experiment (`smoke`, `pilot`, `comparison`) or a path to a YAML file |
| `--runtime NAME` | `setup`, `data`, `prepare`, `start`, `status`, `pull`, `run` | `kaggle`; omit to use this machine |
| `--method`, `--task`, `--scenario NAME` | `prepare`, `status`, `run` | Restrict to these values; each is repeatable |
| `--plugins FILE` | all | A `.py` file or importable module that registers tasks, scenarios or methods; repeatable |
| `--results-dir DIR` | all | Results root (default: `benchmarks/results` in this repository, `./autotunenet_results` elsewhere, or `$AUTOTUNENET_RESULTS`) |

## `run`

Runs every stage for the cells without a result: check the runtime, fetch data, plan, train, collect, report.

| Option | Default | Meaning |
|---|---|---|
| `--device {auto,cpu,cuda,mps}` | `auto` | Device on this machine |
| `--parallel N` | `3` | Kaggle notebooks at once (2 GPUs each) |
| `--no-report` | off | Skip the report at the end |

```bash
autotunenet run --experiment smoke                                   # seconds, toy task
autotunenet run --experiment comparison --method pbt --task cifar10
autotunenet run --experiment comparison --runtime kaggle --parallel 3
autotunenet run --experiment my_experiment.yaml --plugins my_tasks.py
```

The exit code is 0 when every selected cell has a result, and 1 when some remain (run it again to continue). Ctrl-C exits with 130.

## `list`

Prints the registered experiments, tasks, scenarios, methods and runtimes, including those added by `--plugins`.

## `setup`

Checks a runtime and names what is missing. On this machine it prints the device training will use; for Kaggle it checks the CLI and the token. Exits with 1 if the runtime is not ready.

## `data`

```bash
autotunenet data --experiment comparison                 # download what the experiment needs
autotunenet data --task cifar10 --runtime kaggle         # and upload it as a private Kaggle dataset
```

| Option | Meaning |
|---|---|
| `--task NAME` | Tasks to fetch; repeatable. Defaults to the experiment's tasks |
| `--force` | Upload a new version even if the dataset exists |

## `prepare`, `start`, `pull`

The stages of a hosted run, for when you want to inspect the notebooks before anything starts. `prepare` takes `--parallel` and the cell filters. If an earlier round is still running, `prepare` keeps it instead of planning a new one. See [Runtimes](runtimes.md#kaggle).

## `status`

```text
$ autotunenet status --experiment comparison --runtime kaggle
comparison: 84/150 cells done
  asha: 9 to run
  pbt: 21 to run
  ...
kaggle round 1 (autotunenet[bench] @ git+https://github.com/...@4b891a0)
  autotunenet-comparison-1-of-3: running, 22 cells
  autotunenet-comparison-2-of-3: downloaded, 22 cells
```

## `report`

Builds `<results>/<experiment>/report/`: `summary.json`, `support_matrix.{svg,md,html}` and the figures. The matrix shows every planned cell, so unfinished cells appear as pending.

| Option | Meaning |
|---|---|
| `--plan NAME_OR_FILE` | Count another experiment's cells as planned. Use it to show a results folder against a larger plan |
| `--title TEXT` | Matrix title (default `Support matrix`) |

```bash
autotunenet report --experiment comparison
autotunenet report --experiment previous --plan comparison
```

## Environment variables

| Variable | Effect |
|---|---|
| `AUTOTUNENET_RESULTS` | Default results root |
| `AUTOTUNENET_DATA_DIR` | Dataset folder |
| `AUTOTUNENET_LOG_FILE` | Also write the library's log to this file |
