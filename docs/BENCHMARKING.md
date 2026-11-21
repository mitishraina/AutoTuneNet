# Benchmarking

The benchmark harness in [`benchmarks/`](../benchmarks/) compares AutoTuneNet with other ways of choosing the learning rate, under identical conditions: same tasks, models, seeds, epochs, search ranges and held-out test split. This page explains what it measures, how it is organised, and how to extend it.

## What is being compared

AutoTuneNet addresses **online hyperparameter adaptation (OHA)**: adjusting hyperparameters *during* a single training run, without restarts, while training dynamics may change underneath it. That differs from:

- **fixed values and schedules**, which never react to what happens in training;
- **offline HPO** (e.g. Optuna), which searches across many separate training runs;
- **population-based training**, which adapts during training but trains several models at once.

So every result records both **quality** (test accuracy, behaviour after a shift) and **cost** (epochs trained in total, models trained in parallel, restarts, wall time). A method that is slightly more accurate at four times the compute is a different trade-off, not simply better.

## Concepts

| Term | Meaning |
|---|---|
| **Task** | A dataset and model with training defaults: batch size, initial learning rate, and the learning-rate range every tuning method searches (log scale). Built in: `mnist`, `cifar10`, `toy`. |
| **Scenario** | What happens to training besides the method's own decisions: `stationary`, `label_noise` (30% of training labels randomised from epoch 5), `gradient_noise` (noise added to every gradient from epoch 5), `bad_initial_lr` (start at 0.1). |
| **Method** | How the learning rate is chosen (see the table below). |
| **Experiment** | A YAML file listing methods, tasks, scenarios, seeds and epochs. Every combination is one **cell**. A scenario can be limited with `tasks: [...]` or `methods: [...]`. |
| **RunResult** | The JSON every cell produces: test metrics, per-epoch history, compute, environment and config. |

## Methods

| Name | Kind | What it does | Cost per cell (defaults) |
|---|---|---|---|
| `fixed_lr` | baseline | keeps the initial learning rate | 10 epochs |
| `cosine` | schedule | cosine annealing to 0 over the run | 10 epochs |
| `plateau` | reactive rule | halves the LR after 2 epochs in a row without improvement | 10 epochs |
| `autotunenet` | online, single run | Bayesian (TPE) proposals each epoch with a stability guard and rollback | 10 epochs |
| `optuna_tpe` | offline search | 8 full trials, TPE-proposed LRs, best on validation reported | 80 epochs, 7 restarts |
| `asha` | offline search | 16 trials, weak ones stopped early (successive halving) | depends on pruning |
| `hyperband` | offline search | 16 trials with Hyperband early stopping | depends on pruning |
| `pbt` | population | 4 models in parallel; every 2 epochs the weakest copy the best and perturb the LR (Ray Tune) | 40 epochs, 4 in parallel |

Offline search methods sample their own learning rates, so the bad-initial-LR stress test does not apply to them (n/a in the support matrix). In it, PBT members all start from the bad learning rate.

PBT runs its members concurrently, as Ray's PBT is designed to; which results it compares at a synchronisation point can depend on scheduling, so two runs with the same seed can differ slightly. Every other method is bit-reproducible for a given seed and hardware.

## Layout

The harness is part of the package (`src/autotunenet/bench/`), so `pip install "autotunenet[bench]"` gives anyone the same harness and the `autotunenet` command.

```
src/autotunenet/
├── cli.py              # the `autotunenet` command (argparse; one handler per command)
└── bench/
    ├── core/
    │   ├── trainer.py      # Trainer: the one training loop (train, evaluate, test, save/load state)
    │   ├── method.py       # Method / SingleRunMethod / Controller interfaces, RunContext
    │   ├── ledger.py       # ComputeLedger: epochs, parallel models, trials, restarts, time
    │   ├── result.py       # RunResult schema and file layout
    │   ├── experiment.py   # experiment files (built-in by name, or any YAML path), cells
    │   ├── metrics.py      # post-shift metrics, regret (computed from the history)
    │   ├── registry.py     # METHODS, TASKS, SCENARIOS
    │   ├── env.py          # hardware and software details stored with each result
    │   └── device.py       # --device resolution
    ├── tasks/              # one TaskSpec per task
    ├── scenarios/          # one module per scenario
    ├── methods/            # one module per method
    ├── runtimes/           # where cells run: machine.py (this computer), kaggle.py (+ notebook template)
    ├── experiments/        # comparison.yaml (the real run), pilot.yaml, smoke.yaml (toy task, seconds)
    ├── reporting/          # support matrix (data model + SVG/Markdown/HTML) and figures
    ├── run.py              # run cells on this machine, skipping finished ones
    ├── aggregate.py        # mean ± std over seeds -> summary.json
    ├── report.py           # summary + support matrix + figures
    ├── paths.py            # default results and data folders
    └── plugins.py          # --plugins loading
```

The runner only talks to the registries and the `Method` interface, so adding a method, task or scenario never touches `run.py`. A runtime implements the stages `setup`, `data`, `prepare`, `start`, `pull` and `run` (`runtimes/base.py`); adding Colab or a cluster means adding one class to `runtimes/`.

## Running

```bash
autotunenet --help                                         # every command, one line each
autotunenet run --help                                     # one command in detail, with examples
autotunenet list                                           # experiments, tasks, scenarios, methods, runtimes
autotunenet run --experiment comparison                    # everything, on this machine
autotunenet run --experiment comparison --method pbt --task cifar10   # part of it
autotunenet status --experiment comparison                 # cells done, cells left per method
autotunenet report --experiment comparison                 # matrix + figures
autotunenet report --experiment previous --plan comparison # earlier runs, against the full plan
```

`--experiment` takes a built-in name or a path to your own YAML file. Inside a development checkout use `uv run autotunenet ...`.

The support matrix always draws every planned cell: cells without results show as hatched **pending**, cells with only some seeds show "2/3 seeds", and the subtitle counts complete cells, so it doubles as a progress tracker while a run is in flight.

`uv run python benchmarks/legacy_import.py` converts the results of the previous benchmark scripts (fixed LR, cosine, AutoTuneNet; `benchmarks/results/legacy/`) into the `previous` experiment, which the README's matrix shows until the comparison run is complete.

Results go to `<results>/<experiment>/<method>/<task>/<scenario>/seed_<s>.json`, where `<results>` is `benchmarks/results` inside this repository, `./autotunenet_results` anywhere else, or `$AUTOTUNENET_RESULTS`. Each cell is saved as soon as it finishes and finished cells are skipped, so every command can be stopped and run again to continue. Results from several machines merge by copying the folders together.

## Running on Kaggle (your own account)

Kaggle gives free GPU time ("GPU T4 x2", about 30 hours a week). Everything runs under **your** account; no credentials are stored in the repository or in the notebooks.

1. Create a token: kaggle.com → your profile picture → **Settings** → **API** → **Create New Token**, and save it as `~/.kaggle/access_token` (or a downloaded `kaggle.json` in `~/.kaggle/`), then `chmod 600` it. Never commit it or paste it anywhere.
2. Run one command:

```bash
autotunenet run --experiment comparison --runtime kaggle --parallel 3
```

It checks the Kaggle CLI and token, uploads each task's data once as a private dataset `autotunenet-<task>`, writes the notebooks for the cells without results (two GPUs each, `--parallel` notebooks at once), starts them, prints progress every minute, downloads each notebook's results as it finishes and prints the support matrix. Stop it whenever you like: running it again follows the notebooks still running, and once they are done plans a new round with whatever is still missing (failed or timed-out cells).

The same stages one at a time:

```bash
autotunenet setup   --runtime kaggle
autotunenet data    --runtime kaggle --experiment comparison   # --force to upload again
autotunenet prepare --runtime kaggle --experiment comparison --parallel 3
autotunenet start   --runtime kaggle --experiment comparison
autotunenet status  --runtime kaggle --experiment comparison
autotunenet pull    --runtime kaggle --experiment comparison
autotunenet report  --experiment comparison
```

The notebooks install AutoTuneNet from the exact commit you are on when it has been pushed (otherwise the released version), so push first if you changed the harness. They are saved as readable `.ipynb` files, one cell per step, under `<results>/<experiment>/notebooks/`. For the published experiments in this repository they are committed next to the results, so anyone can check what ran; your own experiments, their notebooks and the runtime state (`.runtime/`, which contains your username) are git-ignored or live outside the repository.

## Rules every method follows

- Training uses `Trainer`, so every method sees the same data split, initial weights and shuffling for a given seed.
- Tuning methods search the task's `lr_range` on a log scale.
- Selection uses validation data only. The test split is used once, to report the final model.
- Every epoch of every model counts towards `epochs_trained` in the `ComputeLedger`, including discarded trials.
- Compare results only within one kind of hardware; each result records it, and `aggregate` warns when a comparison mixes hardware.

## Adding a method, task or scenario (plugins)

Put your additions in a Python file and pass it with `--plugins` (repeatable) to any command; nothing in the package needs to change. The same file is shipped to Kaggle with the notebooks.

```bash
autotunenet list --plugins my_tasks.py
autotunenet run --experiment my_experiment.yaml --plugins my_tasks.py [--runtime kaggle]
```

### A method

Most methods train one model and decide something at the end of each epoch. Subclass `SingleRunMethod` and return a `Controller`:

```python
import torch

from autotunenet.bench.core.method import Controller, SingleRunMethod
from autotunenet.bench.core.registry import METHODS


class _StepDecay(Controller):
    def __init__(self, optimizer, step_size, gamma):
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)

    def on_epoch_end(self, epoch, val_loss):
        self.scheduler.step()


@METHODS.register("step_decay")
class StepDecay(SingleRunMethod):
    """Multiply the learning rate by `gamma` every `step_size` epochs."""

    def __init__(self, step_size: int = 3, gamma: float = 0.5):
        self.step_size, self.gamma = step_size, gamma

    def make_controller(self, optimizer, ctx):
        return _StepDecay(optimizer, self.step_size, self.gamma)
```

To make it built in instead, put the module in `src/autotunenet/bench/methods/`, import it in that package's `__init__.py`, add it (with its arguments) to `experiments/smoke.yaml`, and run `uv run pytest`. The end-to-end test fails if a registered method is missing from `smoke.yaml`, and runs it under every scenario on the toy task.

Methods that train several models (offline HPO, population-based training) subclass `Method` and implement `run(ctx, ledger)`. They create models with `ctx.new_trainer()` and `ctx.new_scenario()`, train each candidate with `train_single_run(...)` or the `Trainer` directly, record the cost in the ledger, and return a `RunOutcome` for the model they report.

### A task (any dataset and model)

```python
from autotunenet.bench.core.registry import TASKS
from autotunenet.bench.paths import data_dir
from autotunenet.bench.tasks.base import TaskSpec

TASKS.add("my_task", TaskSpec(
    name="my_task",
    build_loaders=lambda batch_size, seed: ...,   # -> (train_loader, val_loader), split seeded by `seed`
    build_test_loader=lambda batch_size: ...,     # held-out test loader
    build_model=MyModel,
    batch_size=64,
    lr=0.01,
    lr_range=(1e-4, 1e-1),
    data_paths=("my_task",),                      # what it reads, relative to data_dir()
    prepare_data=download_my_task,                # fetch it into data_dir() if missing
))
```

Read data from `data_dir()` (`./data` or `$AUTOTUNENET_DATA_DIR`). `data_paths` and `prepare_data` are what let `autotunenet data --runtime kaggle` upload your data as a private dataset; on Kaggle the data directory is rebuilt from it, so the same loader code works there. For torchvision datasets, `autotunenet.bench.tasks.data.load_dataset` downloads with a progress bar.

### A scenario

Subclass `Scenario` (`autotunenet.bench.scenarios.base`) and register it with `@SCENARIOS.register("name")`, in a plugin file or as a module under `src/autotunenet/bench/scenarios/` (then import it in that package's `__init__.py` and add it to `smoke.yaml`). A scenario sets `shift_epoch` (the first epoch trained under the change, or `None`), can override `initial_lr(task)`, and acts in `on_epoch_start(epoch, trainer)`. If its constructor takes a `seed`, it receives the run's seed automatically.

## Reporting

`autotunenet report` writes to `<results>/<experiment>/report/`:

- `support_matrix.svg`: Fixed LR as the reference row at the top, then one row per method; one column per task and scenario, plus each method's training cost (a log-scale bar of epochs trained, with models in parallel and restarts). Each cell shows the test accuracy (mean ± std over seeds), its difference from Fixed LR in percentage points and its rank in the column (equal scores share a rank). The colour is that difference: teal = better, rose = worse, grey = within ±0.25 pp. Hatched "pending" cells are planned but not run yet; "2/3 seeds" marks a partly done cell; empty "n/a" cells are scenarios that do not apply. Hovering a cell shows its details, and the SVG follows the viewer's light or dark theme. `support_matrix.md` is the same table as Markdown, and `support_matrix.html` a page embedding the SVG.
- `accuracy_vs_compute/`: test accuracy against total epochs trained, per task and scenario.
- `curves/`: validation loss and learning rate per epoch (mean ± std band, shift marked).
- `paper_figures/`: the four figures from the AutoTuneNet paper, per task.

`summary.json` holds, per method, task and scenario, the mean ± std over seeds of:

- test accuracy and loss (held-out split);
- epochs trained, models in parallel, trials, wall time;
- cumulative regret against `fixed_lr` (sum of validation-loss differences per epoch; negative is better);
- post-shift degradation and recovery time, computed from the history against the last pre-shift loss.

Papers or comparisons built on this harness should state the task, scenario, method settings, compute used and number of seeds, and report mean ± std.
