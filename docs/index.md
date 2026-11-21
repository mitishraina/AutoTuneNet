# AutoTuneNet

AutoTuneNet adapts hyperparameters such as the learning rate **during a single training run**. Once per tuning step (usually once per epoch) it scores the last configuration on a validation metric. A Bayesian optimizer (Optuna TPE) proposes the next configuration, which is applied with a bounded change. If the score regresses repeatedly, it rolls back to the last configuration that worked. Training never restarts.

```python
adapter.on_epoch_end(metric=-val_loss)   # the only line added to your training loop
```

## The problem

A learning rate is usually picked before training and then held fixed or decayed on a fixed schedule. Two things go wrong with that:

- **The best value changes during a run.** Early training tolerates large steps and late training needs small ones. A shift in the data (label noise, a new distribution) can change the optimum again mid-run.
- **Finding a good value costs whole runs.** Grid search, random search and offline Bayesian optimization each train many models from scratch, which is impractical for long jobs.

This setting is called **Online Hyperparameter Adaptation (OHA)**: choose $\lambda_t$ at each step $t$ from the history observed so far, inside one run, without restarts. See the [formal definition](oha.md).

## How AutoTuneNet approaches it

| Step | What happens | Why |
|---|---|---|
| Observe | The adapter receives a higher-is-better metric, e.g. `-val_loss` | The controller sees only a scalar, as in any black-box tuner |
| Score | The metric is credited to the configuration that produced it and told to TPE | TPE builds a model of which configurations score well |
| Check | A stability monitor compares the smoothed score with the best so far | One noisy epoch should not trigger a rollback |
| Act | Either accept and ask TPE for the next configuration, clipped to ±`max_delta` of the current one, or restore the last good configuration and start a cooldown | Bounded steps and rollback keep training out of unstable regions |

Details: [How it works](concepts.md).

## Where it fits

| | Offline HPO (grid, random, BO, ASHA) | LR schedules (cosine, step, plateau) | Population-based training | AutoTuneNet |
|---|---|---|---|---|
| Runs trained | Many | One | Several in parallel | One |
| Reacts to metrics | Between trials | No (plateau: yes, one rule) | Yes, by copying better members | Yes, every tuning step |
| Restarts | One per trial | None | Members copy checkpoints | None |
| Safety mechanism | — | — | — | Warmup, bounded updates, rollback, cooldown |

Use AutoTuneNet when one run is all you can afford, when the run is long or the data changes during it, or when you want a learning-rate policy without hand-designing a schedule.

Don't use it when you can afford many full runs and want the best final number (offline HPO wins there), or for architecture choices (layer counts, activations). It tunes numeric optimizer settings only.

## What is in the repository

- **The library** (`autotunenet`): a framework-independent controller (`BayesianOptimizer`, `ParameterSpace`, `StabilityMonitor`) and a PyTorch adapter.
- **The benchmark harness** (`autotunenet.bench`): compares AutoTuneNet with Fixed LR, cosine, ReduceLROnPlateau, Optuna TPE, ASHA, Hyperband and PBT. Tasks, scenarios and seeds are controlled, and compute is recorded for every run.
- **The `autotunenet` command**: runs any benchmark experiment on this machine or on Kaggle, resumably.

## Where to go next

| I want to… | Read |
|---|---|
| Install it | [Installation](installation.md) |
| Add it to my training loop | [Quick start](quickstart.md) |
| Understand the control loop and its guarantees | [How it works](concepts.md) |
| Tune several parameters, use configs | [Configuration](configuration.md) |
| Use hooks, param groups, schedulers | [PyTorch guide](pytorch.md) |
| Look up a class or function | [API reference](api/index.md) |
| Compare methods on my own task | [Benchmarks](benchmarks/index.md) |
| See results so far | [Results](benchmarks/results.md) |
