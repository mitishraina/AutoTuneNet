# API reference

Generated from the docstrings in the source. Everything listed here is public. Anything else (names starting with `_`, and modules not listed) may change between versions.

| Module | Contents |
|---|---|
| [Tuning](tuning.md) | `ParameterSpace`, `BayesianOptimizer`, `Optimizer`, `to_score`, `MetricSmoother` |
| [Safeguards](safeguards.md) | `StabilityMonitor`, `Rollback` |
| [PyTorch adapter](pytorch.md) | `PyTorchHyperParameterAdapter` |
| [Configuration](config.md) | `build_pytorch_autotunenet`, `load_config`, `load_config_from_dict`, `AutoTuneConfig` |
| [Logging](logging.md) | `get_logger`, `enable_file_logging` |
| [Benchmark extension points](bench.md) | `TaskSpec`, `Scenario`, `Method`, `SingleRunMethod`, `Controller`, `RunContext`, registries, `RunResult`, `Runtime` |

Import paths are the module paths shown on each page, e.g. `from autotunenet.parameters import ParameterSpace`.
