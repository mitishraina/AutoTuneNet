<div align="center">

# AutoTuneNet: Online Hyperparameter Adaptation

Bayesian hyperparameter tuning *during* a single PyTorch training run, with bounded updates and rollback.

[![CI](https://github.com/mitishraina/AutoTuneNet/actions/workflows/ci.yml/badge.svg)](https://github.com/mitishraina/AutoTuneNet/actions/workflows/ci.yml)
[![Docs](https://github.com/mitishraina/AutoTuneNet/actions/workflows/docs.yml/badge.svg)](https://mitishraina.github.io/AutoTuneNet/)
[![PyPI](https://img.shields.io/pypi/v/autotunenet.svg)](https://pypi.org/project/autotunenet/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Documentation](https://mitishraina.github.io/AutoTuneNet/) · [Quick start](https://mitishraina.github.io/AutoTuneNet/quickstart/) · [API reference](https://mitishraina.github.io/AutoTuneNet/api/) · [Benchmarks](https://mitishraina.github.io/AutoTuneNet/benchmarks/) · [Citation](#citation)

</div>

<p align="center">
  <img src="docs/assets/architecture.svg" width="100%" alt="AutoTuneNet architecture: the training run sends a validation metric to the controller, which smooths it, checks stability, and either updates a TPE surrogate and applies a bounded hyperparameter change, or rolls back to the last stable configuration." />
</p>

## Abstract

Hyperparameters such as the learning rate are usually fixed before training or follow a predefined schedule, but the best values change during a run, and long jobs are expensive to restart. AutoTuneNet treats tuning as an online control problem inside one run. At each tuning step, a Bayesian optimizer (Optuna TPE, ask–tell) scores the configuration that just ran and proposes the next one. The proposal is applied with a bounded change. A stability monitor rolls back to the last good configuration when the smoothed validation score regresses repeatedly. It attaches to an existing PyTorch loop with one call per epoch. The repository also contains a benchmark harness that compares this against schedules, offline HPO (Optuna, ASHA, Hyperband) and population-based training under controlled shifts, with the compute of every method recorded.

## How it works

Each time you call `adapter.on_epoch_end(metric)`, AutoTuneNet does four things:

1. **Score** the hyperparameters that ran during the last epoch, using your validation metric (e.g. `-val_loss`).
2. **Check stability.** It compares the smoothed score with the best so far. With the defaults, if the score has dropped more than 20% for 2 epochs in a row, it **rolls back** to the last hyperparameters that worked and waits a few epochs before judging again.
3. **Propose** the next hyperparameters with Bayesian optimisation (Optuna TPE), which learns from every epoch scored so far.
4. **Apply them in small steps.** Each value can move by at most ±`max_delta` per epoch (with `max_delta=0.5`, a learning rate of 0.01 can only move to between 0.005 and 0.015), so training never jumps to an extreme setting.

| | Offline HPO | LR schedules | PBT | **AutoTuneNet** |
|---|---|---|---|---|
| Training runs | many | one | several in parallel | **one** |
| Reacts to metrics | between trials | no | yes | **yes, every step** |
| Safety | — | — | — | **bounded steps, rollback, cooldown** |

The full control loop, its guarantees and its limits: [How it works](https://mitishraina.github.io/AutoTuneNet/concepts/).

## Installation

```bash
pip install autotunenet                 # library + `autotunenet` command
pip install "autotunenet[bench]"        # + PBT (Ray Tune), Kaggle runtime, report figures
```

Requires Python 3.10+ and PyTorch. For development: `git clone`, then `uv sync` ([details](https://mitishraina.github.io/AutoTuneNet/installation/)).

## Quick start

```python
import torch
from autotunenet.parameters import ParameterSpace
from autotunenet.bayesian_optimizer import BayesianOptimizer
from autotunenet.adapters.pytorch.adapter import PyTorchHyperParameterAdapter

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

space = ParameterSpace({"lr": {"low": 1e-4, "high": 1e-1, "log": True}})
tuner = BayesianOptimizer(space, seed=0, n_startup_trials=3)
adapter = PyTorchHyperParameterAdapter(optimizer, tuner, warmup_epochs=2, max_delta=0.5)

for epoch in range(epochs):
    train_one_epoch(model, optimizer)
    val_loss = evaluate(model)
    adapter.on_epoch_end(metric=-val_loss)      # higher is better; sets the next epoch's lr
```

A complete script that runs in a second: [`examples/quickstart.py`](examples/quickstart.py). Several parameters, YAML configs, param groups and schedulers are covered in the [user guide](https://mitishraina.github.io/AutoTuneNet/configuration/).

## Benchmarks

<p align="center">
  <img src="benchmarks/results/previous/report/support_matrix.svg" width="100%" alt="Support matrix: test accuracy of AutoTuneNet and seven other learning-rate strategies on MNIST and CIFAR-10 across four scenarios, coloured by the difference from a fixed learning rate; cells not run yet are hatched" />
</p>

Test accuracy on the held-out split, as mean ± std over 3 seeds. Each cell is coloured by its difference from Fixed LR (the top row), and the last column shows each method's training cost. Hatched cells are not run yet. The filled cells are the earlier runs; tables, figures and observations are in [Results](https://mitishraina.github.io/AutoTuneNet/benchmarks/results/).

Reproduce or extend:

```bash
autotunenet run --experiment comparison                       # on this machine
autotunenet run --experiment comparison --runtime kaggle      # on Kaggle GPUs, with your account
autotunenet run --experiment mine.yaml --plugins my_tasks.py  # your own dataset, model or scenario
```

Every command can be stopped and rerun; only unfinished runs train. `autotunenet --help` lists the commands. See the [benchmark guide](https://mitishraina.github.io/AutoTuneNet/benchmarks/).

## Documentation

| | |
|---|---|
| [Installation](https://mitishraina.github.io/AutoTuneNet/installation/) · [Quick start](https://mitishraina.github.io/AutoTuneNet/quickstart/) | Getting it running |
| [How it works](https://mitishraina.github.io/AutoTuneNet/concepts/) · [Configuration](https://mitishraina.github.io/AutoTuneNet/configuration/) · [PyTorch guide](https://mitishraina.github.io/AutoTuneNet/pytorch/) | Using it well |
| [Problem definition (OHA)](https://mitishraina.github.io/AutoTuneNet/oha/) | The formal setting |
| [Benchmarks](https://mitishraina.github.io/AutoTuneNet/benchmarks/) · [CLI](https://mitishraina.github.io/AutoTuneNet/benchmarks/cli/) · [Runtimes](https://mitishraina.github.io/AutoTuneNet/benchmarks/runtimes/) · [Extending](https://mitishraina.github.io/AutoTuneNet/benchmarks/extending/) | Comparing methods |
| [API reference](https://mitishraina.github.io/AutoTuneNet/api/) · [FAQ](https://mitishraina.github.io/AutoTuneNet/faq/) | Lookup |

The same pages are in [`docs/`](docs/) for reading on GitHub.

## Limitations

- **Hyperparameter-only rollback.** Weights and optimizer state are not restored, so damage already done to the weights is not undone.
- **Few observations.** Tuning once per epoch gives TPE about one point per epoch; short runs need a small `n_startup_trials`.
- **Non-stationary objective.** TPE assumes a configuration has one score, but here the score depends on when it is measured. Smoothing, bounded steps and rollback limit the effect, but don't remove it.
- **Coarse control.** Problems inside a tuning interval are seen only at the next call.
- **Small-scale evidence so far.** Results cover small CNNs, 10-epoch runs and 3 seeds, and several differences are within one standard deviation.

## Contributing

Issues and pull requests are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md). AutoTuneNet builds on [Optuna](https://optuna.org/) and [PyTorch](https://pytorch.org/), and the PBT baseline uses [Ray Tune](https://docs.ray.io/en/latest/tune/). Released under the [MIT License](LICENSE).

## Citation

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
