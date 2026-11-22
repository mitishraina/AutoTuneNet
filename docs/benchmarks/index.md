# Benchmarks

The benchmark harness (`autotunenet.bench`, run with the `autotunenet` command) compares AutoTuneNet with other ways of choosing the learning rate under identical conditions: the same tasks, models, seeds, epochs, search ranges and held-out test split. It records the compute each method spends next to its accuracy.

```bash
autotunenet run --experiment smoke          # every method and scenario on a toy task, seconds on a CPU
autotunenet run --experiment comparison     # the full comparison
```

| Page | Contents |
|---|---|
| [Command line](cli.md) | Every `autotunenet` command and option |
| [Runtimes](runtimes.md) | Running on this machine or on Kaggle; resuming |
| [Extending](extending.md) | Your own tasks (any dataset and model), scenarios and methods |
| [Results](results.md) | Published results and how to verify them |

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

## Rules every method follows

- Training uses `Trainer`, so every method sees the same data split, initial weights and shuffling for a given seed.
- Tuning methods search the task's `lr_range` on a log scale.
- Selection uses validation data only. The test split is used once, to report the final model.
- Every epoch of every model counts towards `epochs_trained` in the `ComputeLedger`, including discarded trials.
- Compare results only within one kind of hardware; each result records it, and `aggregate` warns when a comparison mixes hardware.


## Experiment files

An experiment lists what to run. The built-in ones are `smoke` (toy task, seconds), `pilot` (a few cells, for timing) and `comparison` (the published run). Pass a name or a path to your own file:

```yaml
# AutoTuneNet vs. other learning-rate strategies.
#   autotunenet run --experiment comparison
name: comparison
epochs: 10
seeds: [42, 43, 44]
tasks: [mnist, cifar10]

scenarios:
  stationary: {}
  label_noise: {start_epoch: 5, noise_ratio: 0.3}            # 30% of training labels randomised
  gradient_noise: {start_epoch: 5, std: 0.02, tasks: [mnist]} # noise added to every gradient
  # Stress test. Offline search methods have no starting learning rate (each
  # trial samples its own), so it does not apply to them.
  bad_initial_lr:
    initial_lr: 0.1
    methods: [fixed_lr, cosine, plateau, autotunenet, pbt]   # PBT members all start from the bad LR

# Each method's keyword arguments. All tuning methods search the task's lr_range on a log scale.
methods:
  fixed_lr: {}
  cosine: {}
  plateau: {factor: 0.5, patience: 1}                       # halve LR after 2 epochs in a row without improvement
  autotunenet:
    n_startup_trials: 3
    warmup_epochs: 2
    max_delta: 0.5
    smoothing_window: 3
    max_regression: 0.2
    patience: 2
    cooldown: 3
  # Offline search: separate trials, each a full run with its own LR; best on validation is reported.
  optuna_tpe: {n_trials: 8, n_startup_trials: 3}           # 8 full trials
  asha: {n_trials: 16, n_startup_trials: 3}                # stops weak trials early
  hyperband: {n_trials: 16, n_startup_trials: 3}
  # Population-based training (Ray Tune): 4 models in parallel; every 2 epochs the weakest
  # copy the best and perturb the LR (x1.2 / x0.8, or resample).
  pbt: {population: 4, perturbation_interval: 2}
```

| Key | Meaning |
|---|---|
| `name` | Results folder name |
| `epochs` | Epochs per training run (offline methods train each trial for up to this many) |
| `seeds` | One cell per seed |
| `tasks` | Registered task names |
| `scenarios` | Name → parameters; `tasks: [...]` or `methods: [...]` inside limits where it applies |
| `methods` | Name → constructor arguments |

## Outputs and metrics

`autotunenet report` writes to `<results>/<experiment>/report/`:

- `support_matrix.svg`: Fixed LR as the reference row at the top, then one row per method; one column per task and scenario, plus each method's training cost (a log-scale bar of epochs trained, with models in parallel and restarts). Each cell shows the test accuracy (mean ± std over seeds), its difference from Fixed LR in percentage points and its rank in the column (equal scores share a rank). The colour is that difference: teal = better, rose = worse, grey = within ±0.25 pp. Hatched cells with a dash (—) are planned but not run yet, or running now; "2/3 seeds" marks a partly done cell; empty "n/a" cells are scenarios that do not apply. Hovering a cell shows its details, and the SVG follows the viewer's light or dark theme. `support_matrix.md` is the same table as Markdown, and `support_matrix.html` a page embedding the SVG.
- `accuracy_vs_compute/`: test accuracy against total epochs trained, per task and scenario.
- `curves/`: validation loss and learning rate per epoch (mean ± std band, shift marked).
- `paper_figures/`: the four figures from the AutoTuneNet paper, per task.

`summary.json` holds, per method, task and scenario, the mean ± std over seeds of:

- test accuracy and loss (held-out split);
- epochs trained, models in parallel, trials, wall time;
- cumulative regret against `fixed_lr` (sum of validation-loss differences per epoch; negative is better);
- post-shift degradation and recovery time, computed from the history against the last pre-shift loss.

Papers or comparisons built on this harness should state the task, scenario, method settings, compute used and number of seeds, and report mean ± std.

## Code layout

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

