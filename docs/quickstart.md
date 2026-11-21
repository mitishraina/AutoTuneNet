# Quick start

This page adds AutoTuneNet to an existing PyTorch training loop. You need a model, an optimizer and a validation metric; nothing else in the loop changes.

## 1. Define what to tune

```python
from autotunenet.parameters import ParameterSpace

space = ParameterSpace({"lr": {"low": 1e-4, "high": 1e-1, "log": True}})
```

Every name must be a numeric setting of your optimizer's param groups: `lr`, `weight_decay`, `momentum` (SGD), and so on. Use a log-scale range for the learning rate and weight decay.

| Python form | Meaning |
|---|---|
| `(low, high)` | continuous, linear scale |
| `{"low": a, "high": b, "log": True}` | continuous, log scale (`low > 0`) |
| `[a, b, c]` | categorical choices |

## 2. Create the tuner and the adapter

```python
from autotunenet.bayesian_optimizer import BayesianOptimizer
from autotunenet.adapters.pytorch.adapter import PyTorchHyperParameterAdapter

tuner = BayesianOptimizer(space, seed=0, n_startup_trials=3)
adapter = PyTorchHyperParameterAdapter(
    torch_optimizer=optimizer,
    autotune_optimizer=tuner,
    warmup_epochs=2,      # leave the first 2 epochs alone
    max_delta=0.5,        # change each value by at most ±50% per tuning step
)
```

`n_startup_trials` is how many configurations TPE tries at random before it uses its model. Optuna's default of 10 is too many when you tune once per epoch for a few dozen epochs.

## 3. Report a metric once per epoch

```python
for epoch in range(epochs):
    train_one_epoch(model, optimizer)
    val_loss = evaluate(model)
    adapter.on_epoch_end(metric=-val_loss)     # higher is better
```

The adapter writes the next configuration into `optimizer.param_groups` before the next epoch starts. `adapter.last_rollback` is `True` after a call that rolled back.

## A complete, runnable example

[`examples/quickstart.py`](https://github.com/mitishraina/AutoTuneNet/blob/main/examples/quickstart.py) trains a linear model on synthetic data and runs on a CPU in about a second:

```python
--8<-- "examples/quickstart.py"
```

Output (abridged):

```text
epoch  1  val_loss=1.6608  lr=0.01000      # warmup: untouched
epoch  2  val_loss=0.5898  lr=0.00500      # first proposal, clipped to -50%
epoch  5  val_loss=0.0922  lr=0.00322
epoch  9  val_loss=0.0545  lr=0.00020      # TPE moved to small steps as the loss flattened
epoch 14  val_loss=0.0440  lr=0.00153
```

## Logging

Each accepted step and each rollback is logged to the `AutoTuneNet` logger at INFO level. To silence it:

```python
import logging
logging.getLogger("AutoTuneNet").setLevel(logging.WARNING)
```

To also write the log to a file, call `autotunenet.logging.logger.enable_file_logging("autotune.log")` or set the `AUTOTUNENET_LOG_FILE` environment variable.

## Next

- [Configuration](configuration.md): several parameters, stability settings, YAML configs.
- [PyTorch guide](pytorch.md): per-step tuning, param groups, schedulers, checkpoints.
- [How it works](concepts.md): what each setting changes.
