# AutoTuneNet Documentation

AutoTuneNet is a **safe, Bayesian, self-optimizing hyperparameter tuning framework** designed to work inside real training loops.
Unlike traditional hyperparameter search, AutoTuneNet:
- tunes during training
- reacts to metrisc in real time
- protects training with rollback and stability guards
- integrates cleanly with PyTorch


# Who is this for?
- ML engineers tired of manual tuning
- Researchers running long experiments
- Anyone who wants adaptive learning rates and optimzer parameters without fragile hacks


## Key features:
- Bayesian Optimization (Optune-backed)
- Training-time tuning
- Stability Monitoring and rollback
- PyTorch Adapter with hooks
- Multi-parameter support
- Config-driven tuning
- Fully unit tested

## Assumptions and Limitations
AutoTuneNet is designed as a safe, training-time hyperparameter control system, not a general-purpose AutoML framework.

1. Non-stationary Optimization Objective:
AutoTuneNet operates inside a live training loop, where metrics such as loss accuracy naturally evolve over time. As a result:
- The optimization objective is non-stationary
- Early and late training metrics are not globally comparable
- Classical Bayesian Optimization assumptions do not strictly hold

So AutoTuneNet does not treat Bayesian Optimization as a global optimizer.
Instead Bayesian Optimization is used as a proposal mechanism, while safety is enforced externally via stability guards and rollback. The surrogate model is currently:
- not reset
- not explicitly time-decayed
- not reweighted by observation age

This is intentional: surrogate accuracy is not relied upon for safety. Unsafe proposals are rejected by regression detection and rollback. Future versions may explore time-decayed or windowed surrogates, but these are not required for correct behavior.

2. Scope of Hyperparameter Tuning
AutoTuneNet is designed to tune continuous or semi-continuous training hyperparameters, such as:
- learning rate
- momentum
- weight decay
- optimizer coefficients
It does not really:
- select activation functions
- change network depth or width
- modify architecture structure
- perform neural architecture search (NAS)
These problems are discrete, high variance, and unsafe to modify during training.
AutoTuneNet intentionally focuses on training stability and control, not model design.

3. AutoTuneNet prioritizes training safety over aggresive optimization. So exploration is conservative by design.
The goal is not to find a single globally optimal configuration, but to keep training in a stable, high-performing region over time.


### Refer **quickstart.md** for quick guide on usage.