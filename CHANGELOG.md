# Changelog

All notable changes to this project will be documented in this file.
This project follows Semantic Versioning (https://semver.org)

## [v0.1.0] - 2025-12-28

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