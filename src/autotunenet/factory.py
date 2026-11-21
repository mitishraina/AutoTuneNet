"""Build a ready-to-use PyTorch adapter from a configuration dictionary."""
import torch

from autotunenet.parameters import ParameterSpace
from autotunenet.bayesian_optimizer import BayesianOptimizer
from autotunenet.metrics import DEFAULT_SMOOTHING_WINDOW
from autotunenet.adapters.pytorch.adapter import PyTorchHyperParameterAdapter
from autotunenet.config.schema import AutoTuneConfig
from autotunenet.safeguards.stability import StabilityMonitor

def build_pytorch_autotunenet(
    torch_optimizer: torch.optim.Optimizer,
    raw_config: dict,
    seed: int | None = None,
) -> PyTorchHyperParameterAdapter:
    """
    Validate a configuration and build the adapter, optimizer and stability guard from it.

    Every section except `parameter_space` is optional; missing keys take the
    defaults below.

    | Section | Key | Default | Meaning |
    |---|---|---|---|
    | `parameter_space` | *name* | required | `[low, high]` (linear range), `{low, high, log: true}` (log range) or `{choices: [...]}` |
    | `tuning` | `tune_n_steps` | `1` | tune every n calls |
    | `tuning` | `warmup_epochs` | `0` | calls before tuning starts |
    | `tuning` | `warmup_metric_threshold` | `None` | start tuning once the loss (`-metric`) is at or below this value |
    | `tuning` | `max_delta` | `None` | maximum relative change per step, in (0, 1) |
    | `metrics` | `smoothing_window` | `3` | moving-average window |
    | `search` | `n_startup_trials` | `10` | random TPE trials before the model is used |
    | `stability` | `max_regression` | `0.2` | relative drop from the best score counted as a regression |
    | `stability` | `patience` | `2` | consecutive regressions before rollback |
    | `stability` | `cooldown` | `3` | steps with rollback suppressed after one |

    Args:
        torch_optimizer: The `torch.optim.Optimizer` whose hyperparameters are tuned.
        raw_config: The configuration as a dictionary (e.g. loaded from YAML).
        seed: Seed for the TPE sampler.

    Returns:
        A `PyTorchHyperParameterAdapter`; call its `on_epoch_end(metric)` once per epoch.

    Raises:
        ValueError: The configuration is invalid (see `AutoTuneConfig`), or a
            tuned name is not a setting of the optimizer's param groups.

    Examples:
        >>> adapter = build_pytorch_autotunenet(
        ...     optimizer,
        ...     raw_config={
        ...         "parameter_space": {"lr": {"low": 1e-4, "high": 1e-2, "log": True}},
        ...         "tuning": {"warmup_epochs": 2, "max_delta": 0.5},
        ...         "stability": {"patience": 2, "cooldown": 3},
        ...     },
        ...     seed=42,
        ... )
    """

    config = AutoTuneConfig(raw_config)

    param_space = ParameterSpace(config.parameter_space)

    autotune_optimizer = BayesianOptimizer(
        param_space=param_space,
        seed=seed,
        smoothing_window=config.metrics.get("smoothing_window", DEFAULT_SMOOTHING_WINDOW),
        n_startup_trials=config.search.get("n_startup_trials", 10),
    )

    autotune_optimizer.guard = StabilityMonitor(
        max_regression=config.stability.get("max_regression", 0.2),
        patience=config.stability.get("patience", 2),
        cooldown=config.stability.get("cooldown", 3),
    )

    tuning_cfg = config.tuning

    tune_n_steps = tuning_cfg.get("tune_n_steps", 1)
    warmup_epochs = tuning_cfg.get("warmup_epochs", 0)
    warmup_metric_threshold = tuning_cfg.get(
        "warmup_metric_threshold"
    )
    max_delta = tuning_cfg.get("max_delta")


    adapter = PyTorchHyperParameterAdapter(
        torch_optimizer=torch_optimizer,
        autotune_optimizer=autotune_optimizer,
        tune_n_steps=tune_n_steps,
        warmup_epochs=warmup_epochs,
        warmup_metric_threshold=warmup_metric_threshold,
        max_delta=max_delta,
    )

    return adapter


# How users will use the factory:
# adapter = build_pytorch_autotunenet(
#     torch_optimizer,
#     raw_config={
#         "parameter_space": {
#             "lr": [1e-4, 1e-2]              # continuous range
#         },
#         "tuning": {
#             "tune_n_steps": 1,
#             "warmup_epochs": 2,
#             "max_delta": 0.5
#         },
#         "stability": {
#             "max_regression": 0.2,
#             "patience": 2,
#             "cooldown": 3
#         }
#     },
#     seed=42,
# )
# Call once per epoch
# adapter.on_epoch_end(metric=-val_loss)
