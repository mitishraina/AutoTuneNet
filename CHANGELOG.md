# Changelog

All notable changes to this project will be documented in this file.
This project follows Semantic Versioning (https://semver.org)

<!-- --8<-- [start:body] -->
## [Unreleased]

### Added

- `autotunenet` command, installed with the package: `list`, `setup`, `data`, `prepare`, `start`, `status`, `pull`, `report`, `run`. `autotunenet --help` lists them; `autotunenet <command> --help` gives details and examples. Every stage resumes where it stopped.
- Runtimes: this machine (default) and Kaggle (`--runtime kaggle`). The Kaggle runtime uploads data as private datasets, writes readable notebooks, uses two GPUs per notebook and downloads results as notebooks finish.
- `--plugins` for your own tasks (any dataset and model), scenarios and methods; tasks can declare `data_paths` and `prepare_data` for hosted runtimes.
- Benchmark harness (`autotunenet.bench`) with Fixed LR, cosine, ReduceLROnPlateau, AutoTuneNet, Optuna TPE, ASHA, Hyperband and PBT (Ray Tune), a compute ledger per run, and a support matrix (SVG, Markdown, HTML) that tracks progress.
- `pip install "autotunenet[bench]"` extra for PBT, Kaggle and figures.
- Documentation site (MkDocs Material) with an API reference generated from docstrings, published to GitHub Pages; runnable examples in `examples/`.
- `CONTRIBUTING.md`.

### Fixed

- Config parameter spaces: `[low, high]` in YAML/dict configs is now a continuous range. It used to be silently treated as two categorical values. Scientific notation such as `1e-4`, which PyYAML reads as a string, is coerced to float.
- `PyTorchHyperParameterAdapter.from_config` no longer crashes. It had a misspelled `smoothing_window` argument and read a non-existent `config.adapter` section. It now delegates to `build_pytorch_autotunenet`.
- `build_pytorch_autotunenet` now applies the `stability` section (`max_regression`, `patience`, `cooldown`) and accepts a `seed`.
- `autotunenet_bench/run_benchmarks.py` chooses the dataset and model from the config's `task` field (`mnist` | `cifar10`). It previously always used CIFAR-10. The runner also applies the config's stability settings.
- `defaults.yaml` is now included in the built wheel.
- TPE is told the hyperparameters that actually ran, after `max_delta` clipping, not the unclipped proposal.
- The gradient-noise benchmark regime now adds noise on every backward pass. Before, it had no effect on training.
- A NaN/Inf score triggers an immediate rollback and never enters the history. Before, a NaN at the first tuning step disabled the stability guard.
- Rollback restores the last good configuration exactly, instead of clipping it with `max_delta`.
- Configurations rejected by the guard are reported to TPE with their real score, and TPE gets each configuration's own score rather than the smoothed average.
- Regressing steps, during cooldown or within the patience allowance, no longer become the rollback target.
- `step()`, `on_step_end()` and `on_validation_end()` score the configuration that produced the metric, using the same flow as `on_epoch_end()`. The metric warmup gate is latched once passed.
- Post-shift benchmark metrics use a pre-shift baseline that is frozen at the shift.
- The packaged `defaults.yaml` no longer disables tuning (`warmup_metric_threshold` is now 0.5, and values ≤ 0 are rejected).

### Changed

- The benchmark scripts in `benchmarks/` and `autotunenet_bench/` were replaced by `autotunenet.bench` and the `autotunenet` command. Their results are kept in `benchmarks/results/legacy/` and converted to the new format in `benchmarks/results/previous/`.
- Categorical values in configs must be written as `{choices: [...]}`. A continuous range can also be written as `{low: a, high: b}`. Lists that don't have exactly two elements are rejected.
- Config validation now covers `smoothing_window`, `max_regression`, `patience` and `cooldown`.
- Benchmarks are reported as mean ± std over 3 seeds, with test accuracy on the held-out test split, on MNIST and CIFAR-10.
- Development tooling moved to [uv](https://docs.astral.sh/uv/): a `dev` dependency group, a `uv.lock` lockfile, and `.python-version`. CI uses `uv sync --locked`. `requirements.txt` was removed.

### Added

- `CITATION.cff` for GitHub's "Cite this repository".
- `MNISTCNN` model and the `benchmarks/common/tasks.py` task registry.
- Regression tests for config parsing, factory stability settings and `from_config`.
- Log-scale search ranges: `{low, high, log: true}` in configs, and `{"low", "high", "log": True}` in Python. A warning is shown for wide linear ranges.
- `n_startup_trials` (config section `search`) controls how many random trials TPE runs before using its model.
- Benchmark runners take `--task`, `--device` (including Apple MPS) and `--seed`. Adds `autotunenet_bench/aggregate.py` for multi-seed summaries and `plot_results.py`, which also redraws the paper's figures.

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
<!-- --8<-- [end:body] -->
