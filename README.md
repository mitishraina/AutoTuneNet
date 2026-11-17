<div align="center">

# AutoTuneNet: Online Hyperparameter Adaptation

Training-time Bayesian hyperparameter tuning for PyTorch, with rollback-based stability safeguards.

[![CI](https://github.com/mitishraina/AutoTuneNet/actions/workflows/ci.yml/badge.svg)](https://github.com/mitishraina/AutoTuneNet/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Overview](#overview) · [Method](#method) · [Installation](#installation) · [Quick Start](#quick-start) · [Benchmarks](#benchmarks) · [Reproducing Results](#reproducing-results) · [Citation](#citation)

</div>

<p align="center">
  <img src="docs/assets/architecture.svg" width="100%" alt="AutoTuneNet architecture: the training run sends a validation metric to the controller, which smooths it, checks stability, and either updates a TPE surrogate and applies a bounded hyperparameter change, or rolls back to the last stable configuration." />
</p>

---

## Abstract

Hyperparameters such as the learning rate are usually chosen before training and either stay fixed or follow a pre-defined schedule. Training dynamics, however, are non-stationary: the best hyperparameters change over the course of a run, and long-running jobs are expensive to restart. AutoTuneNet treats hyperparameter tuning as an online control problem inside a single training run. A Bayesian optimizer (Optuna TPE, ask–tell) proposes new hyperparameters at each tuning step. Proposals are applied with bounded updates, scored on a smoothed validation signal, and checked by a stability monitor that rolls back to the last known-good configuration when performance regresses consistently. The core is model- and dataset-agnostic and attaches to an existing PyTorch training loop through a single per-epoch call. The repository also includes AutoTuneNet-Bench, a benchmark suite for evaluating Online Hyperparameter Adaptation (OHA) under controlled non-stationarity.

## Overview

| | Offline HPO (grid / random / BO) | LR schedulers | AutoTuneNet (OHA) |
|---|---|---|---|
| When tuning happens | Before training, across many runs | During training, open-loop | During training, closed-loop |
| Restarts required | Yes (one per trial) | No | No |
| Reacts to observed metrics | Between trials only | No (except plateau-based) | Yes, every tuning step |
| Safety against harmful updates | N/A | N/A | Warmup, bounded updates, rollback, cooldown |

AutoTuneNet is:
- a generic, framework-agnostic optimization core with a thin PyTorch adapter,
- a closed-loop controller for hyperparameters (e.g. `lr`, `momentum`, `weight_decay`),
- intended for settings where stability and restart-free training matter more than peak accuracy.

AutoTuneNet is not:
- a model, an AutoML system, or Neural Architecture Search,
- a replacement for offline HPO when you can afford many full runs (see [Benchmark 4](#legacy-benchmarks-benchmarks)),
- population-based training (it operates within a single run).

## Method

AutoTuneNet sits between your training loop and the PyTorch optimizer. Once per tuning step it reads a validation metric, decides whether the last change helped, and writes new hyperparameters into `optimizer.param_groups`.

```mermaid
flowchart LR
    subgraph LOOP["Your training loop"]
        direction TB
        TR["Train one epoch"] --> VA["Validate"]
    end

    subgraph ATN["AutoTuneNet"]
        direction TB
        AD["PyTorchHyperParameterAdapter"] --> SM["MetricSmoother<br/>moving average"]
        SM --> SG{"StabilityMonitor"}
        SG -- "stable" --> BO["BayesianOptimizer<br/>Optuna TPE · ask–tell"]
        SG -- "unstable" --> RB["Rollback<br/>last-good hyperparameters"]
        BO --> CL["Bounded update<br/>±max_delta"]
    end

    VA -- "metric = −val_loss" --> AD
    CL -- "new lr, momentum, …" --> PG[("optimizer.param_groups")]
    RB --> PG
    PG --> TR
```

Decision logic for a single tuning step (`adapter.on_epoch_end(metric)`):

```mermaid
flowchart TD
    A(["on_epoch_end(metric)"]) --> B{"Past warmup and<br/>tuning step due?"}
    B -- no --> Z(["Keep current hyperparameters"])
    B -- yes --> C{"Previous proposal<br/>awaiting a score?"}
    C -- no --> P
    C -- yes --> D["Smooth metric over last w steps"]
    D --> E{"Cooldown active?"}
    E -- yes --> F
    E -- no --> G{"Score NaN/Inf, or dropped more than<br/>max_regression below best, for<br/>patience consecutive steps?"}
    G -- no --> F["Accept<br/>tell TPE the score · mark config as last-good"]
    G -- yes --> R["Roll back<br/>restore last-good config · start cooldown"]
    R --> Z
    F --> P["Propose next config<br/>TPE ask"]
    P --> Q["Clip each value to<br/>current × (1 ± max_delta)"]
    Q --> S(["Write to optimizer.param_groups"])

    classDef accept fill:#e6f4ea,stroke:#2e7d32,color:#1b3d1f
    classDef reject fill:#fdecea,stroke:#c62828,color:#4a1414
    class F accept
    class R reject
```

Design choices:
- **Hyperparameter-only rollback.** Rollback restores the *tuned hyperparameters*, not model weights or optimizer state. This keeps the method cheap and framework-independent. The cost: rollback cannot undo divergence that has already happened in the weights (see [Limitations](#limitations)).
- **Patience and cooldown.** A rollback needs `patience` consecutive regressions, so a single noisy measurement doesn't trigger one. After a rollback, a `cooldown` window suppresses further rollbacks to avoid oscillation.
- **Framework-agnostic core.** `BayesianOptimizer`, `ParameterSpace` and `StabilityMonitor` have no dependency on PyTorch. Only the adapter does.

A formal problem statement for OHA is in [docs/OHA.md](docs/OHA.md).

## Installation

Requires Python ≥ 3.10.

```bash
# From PyPI
pip install autotunenet          # or: uv add autotunenet

# From source (recommended for research / reproducing benchmarks)
git clone https://github.com/mitishraina/AutoTuneNet.git
cd AutoTuneNet
uv sync                          # creates .venv from uv.lock, installs the package + dev tools
```

Development uses [uv](https://docs.astral.sh/uv/). `uv sync` installs the exact versions pinned in `uv.lock`, and `uv run <cmd>` runs a command inside the project environment. If you don't use uv, `pip install -e .` still works.

Dependencies: `optuna`, `torch`, `torchvision`, `pyyaml`.

## Quick Start

```python
import torch
import torch.nn as nn
import torch.optim as optim

from autotunenet.parameters import ParameterSpace
from autotunenet.bayesian_optimizer import BayesianOptimizer
from autotunenet.adapters.pytorch.adapter import PyTorchHyperParameterAdapter

model = nn.Linear(10, 1)
optimizer = optim.Adam(model.parameters(), lr=1e-2)

# Tuple (low, high) = continuous range; list [a, b, c] = categorical choices
param_space = ParameterSpace({"lr": (1e-4, 1e-1)})
autotune = BayesianOptimizer(param_space, seed=42, smoothing_window=3)

adapter = PyTorchHyperParameterAdapter(
    torch_optimizer=optimizer,
    autotune_optimizer=autotune,
    tune_n_steps=1,      # tune every epoch
    warmup_epochs=2,     # no tuning during the first 2 epochs
    max_delta=0.5,       # at most ±50% change per tuning step
)

for epoch in range(20):
    train_one_epoch(model, optimizer)          # your code
    val_loss = evaluate(model)                 # your code
    adapter.on_epoch_end(metric=-val_loss)     # higher is better

    print(f"epoch {epoch:02d} | lr={optimizer.param_groups[0]['lr']:.6f}")
```

The adapter changes `optimizer.param_groups` in place, so the rest of your training loop stays the same. After each call you can check `adapter.last_rollback` to see whether that step triggered a rollback.

### Multi-parameter tuning

```python
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)

param_space = ParameterSpace({
    "lr":           (1e-4, 1e-1),
    "momentum":     (0.80, 0.99),
    "weight_decay": (1e-5, 1e-3),
})
```

Any numeric key in `optimizer.param_groups` can be tuned. Keys that the optimizer doesn't have are ignored.

### Configuring the stability monitor

```python
from autotunenet.safeguards.stability import StabilityMonitor

autotune.guard = StabilityMonitor(
    max_regression=0.2,  # relative drop from best smoothed score counted as a regression
    patience=2,          # consecutive regressions before rollback
    cooldown=3,          # tuning steps with rollback suppressed afterwards
)
```

### Config-driven tuning

```python
from autotunenet.factory import build_pytorch_autotunenet

adapter = build_pytorch_autotunenet(
    torch_optimizer=optimizer,
    raw_config={
        "parameter_space": {"lr": [1e-4, 1e-2]},
        "tuning":    {"tune_n_steps": 1, "warmup_epochs": 2, "max_delta": 0.5},
        "metrics":   {"smoothing_window": 3},
        "stability": {"max_regression": 0.2, "patience": 2, "cooldown": 3},
    },
    seed=42,
)
```

The same config can live in a YAML file:

```yaml
parameter_space:
  lr: [1e-4, 1e-1]                     # continuous range [low, high]
  momentum: {low: 0.8, high: 0.99}     # continuous range, explicit form
  # batch_size: {choices: [32, 64]}    # categorical
tuning:
  tune_n_steps: 1
  warmup_epochs: 2
  max_delta: 0.5
metrics:
  smoothing_window: 3
stability:
  max_regression: 0.2
  patience: 2
  cooldown: 3
```

```python
adapter = PyTorchHyperParameterAdapter.from_config(optimizer, "autotune.yaml", seed=42)
```

In a config, `[low, high]` always means a continuous range, and categorical values must be written as `{choices: [...]}`. In the Python `ParameterSpace` API, a tuple is a range and a list is categorical. See [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

## Hyperparameters of AutoTuneNet

| Parameter | Component | Default | Description |
|---|---|---|---|
| `tune_n_steps` | Adapter | `1` | Tune every *n* calls (epochs or steps) |
| `warmup_epochs` | Adapter | `0` | Epochs before tuning starts |
| `warmup_metric_threshold` | Adapter | `None` | Tuning starts only once `metric ≥ −threshold` |
| `max_delta` | Adapter | `None` | Max relative change per step, in `(0, 1)` |
| `smoothing_window` | Optimizer | `5` (`3` via config) | Moving-average window over observed metrics |
| `max_regression` | StabilityMonitor | `0.2` | Relative regression threshold |
| `patience` | StabilityMonitor | `2` | Consecutive bad steps before rollback |
| `cooldown` | StabilityMonitor | `3` | Steps with rollback suppressed after a rollback |

## Benchmarks

All results are single-seed (seed = 42), 10-epoch runs with a small CNN. They show how the controller behaves (stability, recovery, overhead). They are not state-of-the-art claims.

### AutoTuneNet-Bench: non-stationary training

Setup: Adam optimizer, 30% label noise injected at epoch 5 (`label_noise`, `noise_ratio = 0.3`), except where noted. Controller: `warmup_epochs = 2`, `max_delta = 0.5`, `smoothing_window = 3`, `max_regression = 0.2`, `patience = 2`, `cooldown = 3`. Full protocol: [autotunenet_bench/AUTOTUNENET_BENCH.md](autotunenet_bench/AUTOTUNENET_BENCH.md).

> **Note.** These results were produced before the config parser fix in this release. At that time, a YAML range such as `lr: [1e-4, 1e-1]` was read as two categorical values, so the TPE optimizer could only propose the endpoints, which `max_delta` then clipped. Re-running the configs now searches the full continuous range, so numbers will differ.

| Task | Controller | Final val acc ↑ | Best val acc ↑ | Final val loss ↓ | Post-shift degradation ↓ | Loss spike ↓ | Rollbacks | Overhead / epoch |
|---|---|---|---|---|---|---|---|---|
| MNIST | Fixed LR (1e-2) | 93.68 | 97.60 | 0.596 | 0.454 | 0.531 | 0 | 0.007 ms |
| MNIST | Cosine scheduler | **97.07** | 98.13 | **0.457** | **0.406** | **0.453** | 0 | 0.027 ms |
| MNIST | AutoTuneNet | 96.30 | **98.07** | 0.576 | 0.424 | 0.439 | 1 | 0.67 ms |
| CIFAR-10 | Fixed LR (1e-3) | 70.08 | 70.08 | 1.106 | 0.277 | 0.301 | 0 | 0.007 ms |
| CIFAR-10 | AutoTuneNet | **72.42** | **72.42** | **1.042** | **0.210** | **0.236** | 0 | 1.32 ms |

*MNIST, gradient-noise regime (Gaussian noise σ = 0.02 from epoch 5, AutoTuneNet):* final val acc 96.10, final val loss 0.173, post-shift degradation 0.050, 0 rollbacks.

Cumulative regret vs. fixed LR (sum over epochs of val-loss difference, controller − baseline; negative is better): −0.201 on MNIST and −0.333 on CIFAR-10. AutoTuneNet's validation loss was lower than the fixed-LR baseline's after the distribution shift. ([results/compute_regret/](autotunenet_bench/results/compute_regret/))

### Legacy benchmarks (`benchmarks/`)

Stationary MNIST, Adam, batch size 64, 10 epochs, seed 42. Details in [benchmarks/BENCHMARKS.md](benchmarks/BENCHMARKS.md). The scripts in `benchmarks/` are currently set to CIFAR-10. The stored results below come from their earlier MNIST configuration.

| Benchmark | Method | Final val acc ↑ | Best val acc ↑ | Final val loss ↓ | Restarts |
|---|---|---|---|---|---|
| 1. Baseline | Fixed LR (1e-2) | 97.63 | 97.63 | 0.091 | 0 |
| 2. Scheduler | CosineAnnealingLR | **98.62** | 98.62 | **0.055** | 0 |
| 4. Offline HPO | Optuna, 8 trials × 3 epochs | 98.17 | **99.08** | 0.106 | 8 |
| — | AutoTuneNet (`lr ∈ [1e-4, 1e-1]`, warmup 2, `max_delta` 0.5) | 9.58 | 97.28 | 2.313 | 0 |
| 3. Stress test (initial LR = 0.1) | Fixed LR / Cosine / AutoTuneNet | 9.58 / 11.10 / 9.58 | 11.10 (all) | 2.31 / 2.30 / 2.31 | 0 |

<p align="center">
  <img src="benchmarks/final_results_graph/images/plot_val_accuracy.png" width="45%" alt="Validation accuracy" />
  <img src="benchmarks/final_results_graph/images/plot_lr_history.png" width="45%" alt="Learning-rate history" />
</p>

**Observations.**
1. **Per-step bounds don't stop compounding drift.** On stationary MNIST, AutoTuneNet reached 97.28% but then collapsed to chance. `max_delta = 0.5` limited each step, yet repeated ×1.5 increases drove the LR from 0.01 to ≈0.09 within 8 epochs (see [lr_history.json](benchmarks/autotunenet/results/lr_history.json)). The stability monitor did not intervene before the weights diverged. A tighter search range or a cumulative bound is needed here.
2. **Under distribution shift, closed-loop control helps.** In AutoTuneNet-Bench, AutoTuneNet had lower post-shift degradation and negative cumulative regret against a fixed LR on both MNIST and CIFAR-10.
3. **Schedulers remain strong on stationary problems.** On stationary MNIST, cosine annealing gave the best final metrics, and offline HPO found the best peak configuration at the cost of restarts.
4. **Rollback can't repair lost capacity.** In the stress test every method stays at chance level. Restoring hyperparameters doesn't recover weights that have already diverged.

## Reproducing Results

```bash
# Unit tests
uv run pytest

# AutoTuneNet-Bench: one config = one run; outputs go to autotunenet_bench/results/<config_name>/
# The dataset and model are chosen by the `task` field (mnist | cifar10) in each config.
uv run python -m autotunenet_bench.run_benchmarks --config autotunenet_bench/configs/mnist_fixed_lr.yaml
uv run python -m autotunenet_bench.run_benchmarks --config autotunenet_bench/configs/mnist_scheduler.yaml
uv run python -m autotunenet_bench.run_benchmarks --config autotunenet_bench/configs/mnist_autotunenet.yaml
uv run python -m autotunenet_bench.run_benchmarks --config autotunenet_bench/configs/mnist_gradient_noise.yaml
uv run python -m autotunenet_bench.run_benchmarks --config autotunenet_bench/configs/cifar_fixed_lr.yaml
uv run python -m autotunenet_bench.run_benchmarks --config autotunenet_bench/configs/cifar_autotunenet.yaml
uv run python -m autotunenet_bench.metrics.compute_regret

# Legacy benchmarks
uv run python -m benchmarks.fixed_lr.run
uv run python -m benchmarks.scheduler.run
uv run python -m benchmarks.offline_hpo.run
uv run python -m benchmarks.autotunenet.run
uv run python -m benchmarks.stress_test.run
```

Each AutoTuneNet-Bench run writes `epoch_metric.json`, `stability_metrics.json` and `compute_metrics.json`. All configs set a fixed seed. MNIST and CIFAR-10 are downloaded by `torchvision` into `data/`.

## Repository Structure

```
AutoTuneNet/
├── src/autotunenet/
│   ├── bayesian_optimizer.py   # Optuna-TPE ask–tell optimizer + smoothing + guard
│   ├── parameters.py           # ParameterSpace (continuous ranges / categorical)
│   ├── metrics.py              # Moving-average metric smoother
│   ├── factory.py              # build_pytorch_autotunenet (config → adapter)
│   ├── safeguards/             # StabilityMonitor, Rollback
│   ├── adapters/pytorch/       # PyTorchHyperParameterAdapter + hooks
│   ├── config/                 # Config schema, loader, defaults.yaml
│   └── logging/                # Structured logging and tracking
├── autotunenet_bench/          # AutoTuneNet-Bench: tasks, regimes, controllers, metrics, configs, results
├── benchmarks/                 # Legacy benchmarks: fixed_lr, scheduler, offline_hpo, stress_test, autotunenet
├── docs/                       # Quickstart, configuration, PyTorch guide, OHA definition, FAQ
├── examples/                   # Minimal runnable examples
├── tests/                      # Unit tests (core, safeguards, adapter, config, factory)
├── pyproject.toml              # Package metadata, dependencies, dev dependency group
├── uv.lock                     # Pinned dependency versions
└── CITATION.cff                # Citation metadata (GitHub "Cite this repository")
```

## Limitations

- **Single seed, small models.** The reported numbers come from one seed with lightweight CNNs. Treat them as qualitative evidence.
- **Hyperparameter-only rollback.** Model weights and optimizer state are not restored, so divergence that has already happened is not undone.
- **Epoch-level control.** Tuning happens at a coarse granularity (per epoch or per `tune_n_steps` calls). Instability inside an epoch isn't detected until the epoch ends.
- **Non-stationary objective.** TPE assumes a stationary objective, but here the score of a configuration depends on *when* it is evaluated. Smoothing and bounded updates reduce this mismatch but don't remove it.

## Contributing

Issues and pull requests are welcome. New benchmark regimes, controllers and framework adapters are the most useful contributions. Please open an issue before starting larger changes. New features should come with tests (`uv run pytest`).

## License

This project is released under the [MIT License](LICENSE).

## Acknowledgements

AutoTuneNet builds on [Optuna](https://optuna.org/) (TPE sampler, ask–tell API) and [PyTorch](https://pytorch.org/).

## Citation

If you use AutoTuneNet or AutoTuneNet-Bench in your research, please cite:

```bibtex
@software{raina2025autotunenet,
  author    = {Raina, Mitish},
  title     = {{AutoTuneNet}: Online Hyperparameter Adaptation},
  year      = {2025},
  version   = {1.0.2},
  url       = {https://github.com/mitishraina/AutoTuneNet},
  license   = {MIT}
}
```
