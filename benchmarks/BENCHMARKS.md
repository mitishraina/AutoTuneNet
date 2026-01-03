# AutoTuneNet Benchmarks

This document describes the benchmark experiments used to evaluate AutoTuneNet's behavior, safety, and pratical usefulness.
These benchmarks are designed to answer **trust and adoption questions**, not to clain state-of-the-art(sota) performance.

Introducing a warmup phase and bounded update constraints (`max_delta`) significantly improves AutoTuneNet's reliability. These safeguards prevent catastrophic early exploration while preserving adaptivity, allowing AutoTuneNet to remain competitive with standard schedulers and outperform fixed hyperparameters in stress scenarios.

## Goals of Benchmarking
The benchmarks aim to answer three questions majorly:
1. **Safety**: Does AutoTuneNet avoid destabilizing training?
2. **Reasonableness**: Does it behave comparably to standard practice?
3. **Scope Clarity**: When should AutoTuneNet be used to instead of schedulers or offline HPO?

## General Setup

All benchmarks share the same core setup unless stated otherwise.
- Dataset: **MNIST**
- Model: Simple CNN (2 conv layers + 2 FC layers)
- Optimizer: SGD or Adam (as specified)
- Training epochs: Fixed across all methods
- Random seeds: Fixed per experiment
- Hardware: Single GPU or CPU (results are relative, not absolute)

Only the **hyperparameter strategy** differs between runs.

## Benchmark 1: Fixed Learning Rate vs AutoTuneNet
### Purpose
Establish baseline trust.
This benchmark answers:
> "If i already use a reasonable fixed learning rate, does AutoTuneNet make things worse?"

### Compared Methods
- Fixed Learning rate (hand-picked reasonable value)
- AutoTuneNet (training-time Bayesian tuning)

### Metrics Reported
- Validation loss curve
- Final validation loss
- Number of rollback events

### Expected Interpretation
- AutoTuneNet should not diverge
- Performance should be comparable or slightly better
- Stability is more important than marginal gains


## Benchmark 2: Learing Rate scheduler vs AutoTuneNet
### Purpose
Compare against common real-world practice. Schedulers are the primary alternative most users consider.

### Compared Methods
- Fixed Learning rate
- CosineAnnealingLr
- ReduceLRonPlateau (optional for now)
- AutoTuneNet

### Metrics Reported
- Validation loss curve
- Sensitivity to initial learning rate
- Stability events (rollbacks, oscillations)

### Expected Interpretation
- Schedulers are open-loop heuristics
- AutoTuneNet is closed-loop and adapts to observed behavior

This benchmark demonstrates when adaptive control is useful.


## Benchmark 3: Stress Test (Bad Initial Learning Rate)
### Purpose
Demonstrate safety under failure conditions.
This benchmark intentionally starts training in an unstable regime.

### Setup
- Initial learning rate set significantly higher than recommended
- Same model and data as other benchmarks

### Compared Methods
- Fixed Learning rate
- CosineAnnealingLr
- AutoTuneNet

### Metrics Reported
- Validation loss trajectory
- Training divergence and NaNs
- Recovery behavior after instability

### Expected Interpretation
AutoTuneNet should:
- detect instability
- rollback unsafe updates
- recover and continue training

This benchmark highlights AutoTuneNet's safety guarantees.

## Benchmark 4: Offline Hyperparameter Optimization(HPO) vs AutoTuneNet
### Purpose
Clarify scope and intended use.

This benchmark answers:
> "Is AutoTuneNet replacing offline hyperparameter search?"

### Compared Methods
- Fixed Learning rate
- Offline Optuna (limited trial budget)
- AutoTuneNet (single continous run)

### Constraints
- Equalized total training budget
- No additional restarts for AutoTuneNet

### Expected Interpretation
Offline HPO may find better final configurations. AutoTuneNet trades global optimality for:
- continuous adaptation
- safety
- no restarts

These approaches solve different problems.

### Key Observations
1. **Safety-first behavior**
- AutoTuneNet avoids catastrophic divergence even under adversarial conditions.
- Warmup and bounded updates prevent destructive early exploration.

2. **Closed-loop adaptation**
- Unlike schedulers, AutoTuneNet adapts based on observed metrics.
- Hyperparameter changes are reactive, not predefines.

3. **Failure Containment**
- In stress test with unsafe initial learning rates, AutoTuneNet prevents runaway degradation.
- Once model capacity is lost, AutoTuneNet does not claim recovery, but guarantees bounded behavior.

2. **Comparison to baselines**
- Fixed hyperparameters are brittle to poor initialization.
- Schedulers are stable but inflexible
- Offline HPO achieves strong final metrics but requires starts
- AutoTuneNet trades peak performance for adaptability and safety within a single run

## Important Notes
- The benchmarks do not claim **state-of-the-art** results.
- Results are intended to be interpreted qualitatively.
- AutoTuneNet prioritizes stability and safety over aggresive optimization.
- In stress scenarios where aggressive hyperparameter updates corrupt model weights, AutoTuneNet prevents further destabilization but does not restore lost model capacity. This behavior is intentional and highlights the difference between online control and restart-based optimization.

## Reproducibility
- Uses fixed seeds
- reports configuration files
- can be reproduced using provided scripts
