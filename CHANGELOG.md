# Changelog

All notable changes to this project will be documented in this file.
This project follows Semantic Versioning (https://semver.org)

## [Unreleased]

### Fixed
- Config parameter spaces: `[low, high]` in YAML/dict configs is now a continuous range. It used to be silently treated as two categorical values. Scientific notation such as `1e-4`, which PyYAML reads as a string, is coerced to float.
- `PyTorchHyperParameterAdapter.from_config` no longer crashes. It had a misspelled `smoothing_window` argument and read a non-existent `config.adapter` section. It now delegates to `build_pytorch_autotunenet`.
- `build_pytorch_autotunenet` now applies the `stability` section (`max_regression`, `patience`, `cooldown`) and accepts a `seed`.
- `autotunenet_bench/run_benchmarks.py` chooses the dataset and model from the config's `task` field (`mnist` | `cifar10`). It previously always used CIFAR-10. The runner also applies the config's stability settings.
- `defaults.yaml` is now included in the built wheel.

### Changed
- Categorical values in configs must be written as `{choices: [...]}`. A continuous range can also be written as `{low: a, high: b}`. Lists that don't have exactly two elements are rejected.
- Config validation now covers `smoothing_window`, `max_regression`, `patience` and `cooldown`.
- Development tooling moved to [uv](https://docs.astral.sh/uv/): a `dev` dependency group, a `uv.lock` lockfile, and `.python-version`. CI uses `uv sync --locked`. `requirements.txt` was removed.

### Added
- `CITATION.cff` for GitHub's "Cite this repository".
- `MNISTCNN` model and the `benchmarks/common/tasks.py` task registry.
- Regression tests for config parsing, factory stability settings and `from_config`.

## [v1.0.1] - 2025-12-28

### Added 
- Bayesian hyperparameter optimization using Optuna (ask-tell interface)
- Training-time hyperparameter tuning (no separate trials)
- Stability monitoring with:
    - regression detection
    - consecutive failure thresholds (patience)
    - cooldown periods
- Safe rollback to last known good hyperparameters
- Metric smoothing for noisy training signals
- PyTorch adapter for seamless integration with training loops
- Adapter hooks: `on_step_end`, `on_epoch_end`, `on_validation_end`
- Multi-parameter tuning support (eg. lr, momentum, weight_decay)
- Config-driven tunning with file + console output
- Full unit test coverage for:
    - core optimizer
    - stability safeguards
    - PyTorch adapter
    - config loader

### Changed
- Optimizer lifecycle stabilized to prevent double-tell and state corruption
- Logging improved to reduce noise and highlight state transitions

### Fixed 
- Optuna trial lifecycle errors
- Repeated rollback log spam
- Unsafe Optimizer state mutation during tuning

### Notes
- This is the first public release of AutoTuneNet
- API is stable for core usage, but subject to refinement based on community feedback.


## [v1.0.2] - 2026-01-03

### Added
- Safe warmup phase for training-time hyperparameter tuning(supports epoch-based & metric-based warmup)
- Bounded update constraint (`max_delta`) to prevent aggresive hyperparameter jumps
- Config-driven exposure of tuning safety parameters
- Factory based adapter construction for PyTorch
- Comprehensive benchmark suite with reproducible scripts

### Improved
- Training stability under non-stationary objectives
- Benchmark reliability and reproducibility
- Default out-of-the-box behaviour
- Documentation clarity around assumptions and limitations

### Benchmarks
- Added comparison against:
    - Fixed Learning rate
    - CosineAnnealing LR
    - Offline Optuna-based HPO
- Added stress tests with intentionally unsafe initial configurations
- Included plots for loss, accuracy, learning rate evolution

### Notes
AutoTuneNet v1.0.2 prioritizes safe and predictable adaptation over aggresive optimization, making it suitable for long-running and sensitive training workflows.