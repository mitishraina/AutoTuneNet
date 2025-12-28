# Configuration

AutoTuneNet supports YAML-based configuration for reproducibility.

## Example Config
```yaml
parameter_space:
  lr: [0.001, 0.1]
  weight_decay: [0.0001, 0.01]

tuning:
  tune_n_steps: 1

metrics:
  smoothing_window: 5

stability:
  max_regression: 0.2
  patience: 2
  cooldown: 3
```

## Using Config
```bash
adapter = PyTorchHyperparameterAdapter.from_config(
    torch_optimizer = optimizer,
    config_path = "autotune.yaml"
)
```
Configs allow:
- experiment versioning
- easy sharing
- reproducibilty

