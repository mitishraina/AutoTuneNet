### AutoTuneNet-Bench

AutoTuneNeta-Bench is a standarized benchmark suite for evaluating online hyperparameter adaptation methods under non-stationary training dynamics, with explicit consideration of training stabiliity and compute cost.
Unlike traditional hyperparameter optimization benchmarks that assume independent, restartable training runs, AutoTuneNet-Bench focuses on single run adaptation, where hyperparameters must be adjust safely during an ongoing training process.

### Motivation
Most HPO methods operate offline, requiring multiple full training runs and assuming a stationary objective. In practice, training dynamics are non-stationary, long-running jobs are expensive to restart, and instability can lead to catastrophic failure.

AutoTuneNet-Bench is designed to answer:
```
How well can a method adapt hyperparameters online, within a single training run, under non-stationary conditions, without restarting training?
```
The benchmark emphasizes robustness, safety and compute efficiency, rather than raw final accuracy.

### Problem Definition: Online Hyperparameter Adaptation (OHA)
The benchmark formalizes the task of OHA as follows:
- Training proceeds in a single, continouos run
- Hyperparameters may be adapted at discrete time steps
- Training dynamics may change over time (non-stationarity)
- Training must not be restarted
- Adaptation is subject to stability and compute constraints

This setting differs fundamentally from offline HPO, learning rate schedules, and population based training (PBT).

```graphql
autotunenet_bench/
├── tasks/              # Benchmark training tasks
├── regimes/            # Non-stationarity definitions
├── controllers/        # Baseline and reference controllers
├── metrics/            # Evaluation metrics
├── configs/            # Reproducible experiment configs
├── runners/            # Benchmark execution logic
└── results/            # Logged outputs and plots
```

Each benchmark run is defined by the tuple:
```bash
(task, non_stationarity_regime, controller, config)
```

### Benchmark Tasks (v1)

## MNIST-CNN (Stationary)
- Standard CNN on MNIST
- No non-stationarity
- Used as a sanity check

## MNIST-CNN (Poor Initialization)
- Deliberately suboptimal initial hyperparameters
- Evaluates recovery and robustness

```Note
NOTE: v1 intentionally uses lightweight models to isolate control behavior from architectural complexity.
```

### Non-Stationarity Regimes
Non-stationarity is explicit, controllable, and reproducible.

Each regime implements the interface:
```python
apply(epoch, model, data, optimizer)
```

## Requrired Regimes
- Data Shift: Introduces label noise or distribution change mid-training.
- Curriculum Shift: Increases task difficulty at a fixed epoch.
- Optimizer Sensitivity Shift: Changes optimizer dynamics (eg., regularization strength).
- Loss Perturbation (Stress Test): Injects noise into the loss signal to test robustness.

Each regime is parameterized by:
- trigger time
- severity
- random seed

### Controllers (Baselines)
Controllers determine how hyperparameters are adapted online.

## Controller Interface:
```bash
class Controller:
    def observe(metric, step): ...
    def propose(current_params): ...
```

## Required Controllers (v1)
Baselines
- Fixed hyperparameters
- Learning rate scheduler (eg., ReduceLROnPlateau)

Offline
- Offline Bayesian HPO (restart-based)

Online 
- Random online adaptation
- Heuristic online controller
- AutoTuneNet (safe online controller)

```Note
IMPORTANT: Population-based methods are intentionally excluded in v1 to preserve the single run constraint
```

### Evalutation Metrics
AutoTuneNet-Bench prioritizes behavioral and compute-aware evaluation.

## Performance Metrics
- Final validation accuracy
- Final validation loss

## Stability Metrics
- Divergence events
- Loss spikes
- Rollback count
- Metric vairance across seeds

## Adaptation Metrics
- Hyperparameter update magnitude
- Adaptation frequency
- Recovery time after non-stationary shocks

## Compute Metrics
- Number of training restarts(must be zero)
- Number of optimizer evaluations
- Wall clock time
- Approximate FLOPs

```
Compute adjusted performance is considered more important than raw accuracy.
```

### Reproducibility Protocol
All benchmark runs must specify:
- Fixed random seed
- Deterministic config file
- Explicit non-stationarity regime

```bash
python run_benchmark.py --config configs/mnist_data_shift.yaml
```

### Reporting Standards
Papers using AutoTuneNet-Bench are expected to report:
- task and non-stationarity regime
- controller type
- compute budget
- stability metrics
- results averaged over multiple seeds
This ensures fair and reproducible comparison.

Research Questions Enabled
AutoTuneNet-Bench enables systematic study of:
- Stability vs. performance trade-offs
- Compute efficiency of online adaptation
- Robustness under non-stationary training
- Minimal controller complexity for safe adaptation
- Failure modes of offline hyperparameter optimization

### Intended Use
AutoTuneNet-Bench is designed for:
- researchers studying online optimization and training stability
- systems-oriented ML research
- benchmarking adaptive training methods under realistic constraints
It is not intended as a leaderboard for maximizing accuracy.

### Citation
If you use AutoTuneNet-Bench in your research, please cite:
```bash
AutoTuneNet: Safe Online Hyperparameter Adaptation
```