# Configuration

AutoTuneNet can be configured from a YAML file or a Python dict. Both go through the same schema (`autotunenet.config.schema.AutoTuneConfig`), and invalid values raise `ValueError` when the adapter is built.

## Example config

```yaml
parameter_space:
  lr: {low: 1e-4, high: 1e-1, log: true}   # continuous range, log scale
  weight_decay: {low: 1e-5, high: 1e-3, log: true}
  # momentum: [0.8, 0.99]                  # continuous range, linear scale
  # batch_size: {choices: [32, 64]}    # categorical

tuning:
  tune_n_steps: 1
  warmup_epochs: 2
  warmup_metric_threshold: 0.5         # optional
  max_delta: 0.5

search:
  n_startup_trials: 3                  # random proposals before TPE uses its model

metrics:
  smoothing_window: 3

stability:
  max_regression: 0.2
  patience: 2
  cooldown: 3
```

## Parameter space syntax

| Form | Meaning |
|---|---|
| `name: [low, high]` | Continuous range. The list must have exactly two numbers. |
| `name: {low: a, high: b}` | Continuous range (explicit form). |
| `name: {low: a, high: b, log: true}` | Continuous range sampled on a log scale (needs `low > 0`). Use this for learning rate and weight decay. |
| `name: {choices: [v1, v2, ...]}` | Categorical choices. |

Numbers in scientific notation such as `1e-4` are accepted. PyYAML reads them as strings, so the schema coerces them to floats. A list with more or fewer than two values is rejected, so that a categorical intent is never silently treated as a range.

A linear range spanning two or more orders of magnitude (for example `[1e-4, 1e-1]`) puts about 90% of samples in the top decade. AutoTuneNet warns when it sees one; use `log: true` instead.

In the Python `ParameterSpace` API used directly (not through a config), a **tuple** `(low, high)` is a linear range, `{"low": a, "high": b, "log": True}` is a log-scale range, and a **list** is categorical.

## Options

| Section | Key | Default | Constraint | Description |
|---|---|---|---|---|
| `tuning` | `tune_n_steps` | `1` | int > 0 | Tune every *n* adapter calls |
| `tuning` | `warmup_epochs` | `0` | int ≥ 0 | Epochs before tuning starts |
| `tuning` | `warmup_metric_threshold` | `None` | number | Tuning starts only once `metric ≥ −threshold` |
| `tuning` | `max_delta` | `None` | 0 < x < 1 | Max relative change per tuning step |
| `search` | `n_startup_trials` | `10` | int ≥ 0 | Trials TPE samples at random before using its model. Runs with few tuning steps need a small value (e.g. 3), or every proposal is random |
| `metrics` | `smoothing_window` | `3` | int > 0 | Moving-average window over observed metrics |
| `stability` | `max_regression` | `0.2` | number ≥ 0 | Relative drop from best smoothed score counted as a regression |
| `stability` | `patience` | `2` | int > 0 | Consecutive regressions before rollback |
| `stability` | `cooldown` | `3` | int ≥ 0 | Tuning steps with rollback suppressed after a rollback |

## Using a config

From a YAML file:

```python
from autotunenet.adapters.pytorch.adapter import PyTorchHyperParameterAdapter

adapter = PyTorchHyperParameterAdapter.from_config(
    torch_optimizer=optimizer,
    config_path="autotune.yaml",
    seed=42,
)
```

From a dict:

```python
from autotunenet.factory import build_pytorch_autotunenet

adapter = build_pytorch_autotunenet(optimizer, raw_config={...}, seed=42)
```

The packaged defaults live in [`src/autotunenet/config/defaults.yaml`](../src/autotunenet/config/defaults.yaml).

Configs make it easy to version experiments, share them, and reproduce runs.
