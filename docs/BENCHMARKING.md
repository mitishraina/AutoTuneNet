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

```
benchmarks/
├── core/
│   ├── trainer.py      # Trainer: the one training loop (train, evaluate, test, save/load state)
│   ├── method.py       # Method / SingleRunMethod / Controller interfaces, RunContext
│   ├── ledger.py       # ComputeLedger: epochs, parallel models, trials, restarts, time
│   ├── result.py       # RunResult schema and file layout
│   ├── experiment.py   # experiment files, cells, sharding
│   ├── metrics.py      # post-shift metrics, regret (computed from the history)
│   ├── registry.py     # METHODS, TASKS, SCENARIOS
│   ├── env.py          # hardware and software details stored with each result
│   └── device.py       # --device resolution
├── tasks/              # one TaskSpec per task
├── scenarios/          # one module per scenario
├── methods/            # one module per method
├── experiments/        # comparison.yaml (the real run), smoke.yaml (toy task, seconds)
├── reporting/          # support matrix (data model + SVG/Markdown/HTML) and figures
├── run.py              # CLI: run an experiment, or one shard of it
├── aggregate.py        # mean ± std over seeds -> summary.json
├── report.py           # summary + support matrix + figures
└── download_data.py    # fetch datasets ahead of time
```

The runner only talks to the registries and the `Method` interface, so adding a method, task or scenario never touches `run.py`.

## Running

```bash
uv run python -m benchmarks.run --experiment benchmarks/experiments/comparison.yaml   # everything
uv run python -m benchmarks.run --experiment ... --list                               # what would run
uv run python -m benchmarks.run --experiment ... --shard 2/4                          # a quarter of the cells
uv run python -m benchmarks.aggregate --experiment comparison                         # summary table
uv run python -m benchmarks.report --experiment comparison                            # matrix + figures
uv run python -m benchmarks.report --experiment comparison \
    --plan benchmarks/experiments/comparison.yaml                                      # ... with progress
```

With `--plan`, the support matrix draws every planned cell: cells without results show **—** (not run yet), cells with only some seeds show "2/3 seeds", and the subtitle counts complete cells, so the matrix doubles as a progress tracker while a run is in flight (for example across Kaggle shards).

`uv run python -m benchmarks.legacy_import` converts the results of the previous benchmark scripts (fixed LR, cosine, AutoTuneNet; `benchmarks/results/legacy/`) into the `previous` experiment, which the README's matrix shows until the comparison run is complete.

Results go to `benchmarks/results/<experiment>/<method>/<task>/<scenario>/seed_<s>.json`. Finished cells are skipped, so a run can be interrupted and restarted, or split across machines with `--shard` and merged by copying the result folders together. Datasets download automatically on first use (see the README).

## Rules every method follows

- Training uses `Trainer`, so every method sees the same data split, initial weights and shuffling for a given seed.
- Tuning methods search the task's `lr_range` on a log scale.
- Selection uses validation data only. The test split is used once, to report the final model.
- Every epoch of every model counts towards `epochs_trained` in the `ComputeLedger`, including discarded trials.
- Compare results only within one kind of hardware; each result records it, and `aggregate` warns when a comparison mixes hardware.

## Adding a method

Most methods train one model and decide something at the end of each epoch. Subclass `SingleRunMethod` and return a `Controller`:

```python
# benchmarks/methods/step_decay.py
import torch

from benchmarks.core.method import Controller, SingleRunMethod
from benchmarks.core.registry import METHODS


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

Then import it in `benchmarks/methods/__init__.py`, add it (with its arguments) to `benchmarks/experiments/smoke.yaml`, and run `uv run pytest`. The end-to-end test fails if a registered method is missing from `smoke.yaml`, and runs it under every scenario on the toy task.

Methods that train several models (offline HPO, population-based training) subclass `Method` and implement `run(ctx, ledger)`. They create models with `ctx.new_trainer()` and `ctx.new_scenario()`, train each candidate with `train_single_run(...)` or the `Trainer` directly, record the cost in the ledger, and return a `RunOutcome` for the model they report.

## Adding a task

Register a `TaskSpec` in `benchmarks/tasks/__init__.py`:

```python
TASKS.add("fashion_mnist", TaskSpec(
    name="fashion_mnist",
    build_loaders=lambda batch_size, seed: ...,   # -> (train_loader, val_loader)
    build_test_loader=lambda batch_size: ...,     # held-out test loader
    build_model=MyModel,
    batch_size=64,
    lr=0.01,
    lr_range=(1e-4, 1e-1),
))
```

Load torchvision datasets with `benchmarks.tasks.data.load_dataset` so they download with progress into the shared data folder.

## Adding a scenario

Subclass `Scenario` in a new module under `benchmarks/scenarios/`, register it, import it in `benchmarks/scenarios/__init__.py`, and add it to `smoke.yaml`. A scenario sets `shift_epoch` (the first epoch trained under the change, or `None`), can override `initial_lr(task)`, and acts in `on_epoch_start(epoch, trainer)`. If its constructor takes a `seed`, it receives the run's seed automatically.

## Reporting

`report.py` writes to `benchmarks/results/<experiment>/report/`:

- `support_matrix.svg`: one row per method, one column per task and scenario, plus the cost of each method. Each cell is the test accuracy (mean ± std over seeds) with its rank in the column. Colour codes: the cell shade is the rank (lightest = best; equal scores share a rank), a red triangle means worse than Fixed LR in that column, grey n/a means the scenario does not apply, and Fixed LR is the dashed baseline row. Hovering a cell shows its details. `support_matrix.md` is the same table as Markdown, and `support_matrix.html` a page embedding the SVG.
- `accuracy_vs_compute/`: test accuracy against total epochs trained, per task and scenario.
- `curves/`: validation loss and learning rate per epoch (mean ± std band, shift marked).
- `paper_figures/`: the four figures from the AutoTuneNet paper, per task.

`aggregate.py` reports, per method, task and scenario, the mean ± std over seeds of:

- test accuracy and loss (held-out split);
- epochs trained, models in parallel, trials, wall time;
- cumulative regret against `fixed_lr` (sum of validation-loss differences per epoch; negative is better);
- post-shift degradation and recovery time, computed from the history against the last pre-shift loss.

Papers or comparisons built on this harness should state the task, scenario, method settings, compute used and number of seeds, and report mean ± std.
