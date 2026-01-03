<img width="1564" height="782" alt="image" src="https://github.com/user-attachments/assets/c66061e7-7e4f-42e4-94dd-a4831474409b" />


# AutoTuneNet

A generic, open-source Python library that enables **self-optimizing model training** by dynamically tuning hyperparameters during training using **Bayesian Optimization**.

Instead of traditional manually tuning learning rates, batch sizes, or regularization values before training, AutoTuneNet continuously observes training behavior and automatically adjusts hyperparameters to improve convergence and performance.
**AutoTuneNet combines Bayesian optimization with safety-aware adaptation to tune hyperparameters during training without destabilizing optimization. It is designed for scenarios where stability and restart free training matter.**

## Why This Project Exists

Hyperparameter tuning is one of the most time-consuming and error-prone parts of machine learning workflows.

Common problems:
- Manual trial-and-error
- Grid/random search waste compute
- Hyperparameters are fixed before training
- Optimal values often change during training

**AutoTuneNet solves this by making hyperparameter tuning part of the training loop itself.**


## Core Idea

AutoTuneNet treats hyperparameter tuning as a **learning problem**.

During training:
1. The model trains normally
2. Training and validation metrics are observed
3. A Bayesian optimizer models the relationship between hyperparameters and performance
4. Hyperparameters are updated **incrementally and safely**
5. Training continues with improved settings

This creates a closed-loop, self-optimizing training system.


## What This Is (and Is Not)

### This project is
- A **generic hyperparameter optimization engine**
- **Model-agnostic**
- **Dataset-agnostic**
- Designed to plug into existing training loops
- Suitable for research and production workflows

### This project is not
- A single ML model
- Offline AutoML that runs many full trials
- Grid or random search
- Neural Architecture Search


## Design Philosophy

- **Framework-agnostic core**  
  The Bayesian optimization logic does not depend on PyTorch or TensorFlow.

- **Thin framework adapters**  
  Framework-specific code lives in adapters (PyTorch first).

- **Safety first**  
  Guardrails prevent unstable updates and allow rollback.

- **Minimal user code changes**  
  Users should be able to integrate this with a few lines of code.


## Key features
- Training time hyperparameter optimization
- Bayesian Optimization (Optuna-backed, ask-tell)
- Stability guards with rollback protection
- Metric Smoothing for noisy signals
- PyTorch Adapter
- Multi-parameter tuning(lr, momentum, weight_decay etc.)
- Config-driven tuning via YAML or dict
- Fully Unit Tested
- Lightweight & Modular

## Installation
```bash
pip install autotunenet
```
## Quick Usage
```bash
import torch 
import torch.nn as nn
import torch.optim as optim

from autotunenet.core.parameters import ParameterSpace
from autotunenet.core.bayesian_optimizer import BayesianOptimizer
from autotunenet.adapters.pytorch.adapter import PyTorchHyperparameterAdapter

model = nn.Linear(10, 1)
optimizer = optim.Adam(model.parameters(), lr=0.01)

param_space = ParameterSpace({
    "lr": (1e-4, 1e-1)
})

autotune = BayesianOptimizer(param_space)

adapter = PyTorchHyperparameterAdapter(
    torch_optimizer=optimizer,
    autotune_optimizer=autotune
)

for epoch in range(20):
    train_loss = train_one_epoch(model)
    val_metric = -train_loss  # higher is better

    adapter.on_epoch_end(metric=val_metric)

    print(f"Epoch {epoch} | lr={optimizer.param_groups[0]['lr']:.6f}")
```
That's it
AutoTuneNet will:
- explore hyperparameters
- keep the best configuration
- rollback unsafe updates automatically

## How it works?
AutoTuneNet runs a suggest -> observe loop inside training.
1. Suggest new hyperparameters (Bayesian optimization)
2. Apply them tentatively
3. Observe training or validation metric
4. Accept or rollback based on stability rules

This loop repeats throughout training without breaking it.

## Safety and Stability and Support
AutoTuneNet is designed to never destabilize training.

- Built-in protections:
- Regression detection
- Consecutive failure thresholds(patience)
- Cooldown after rollback
- Restore last known good configuration
- Warmup phase
- Bounded Updates

If a suggested hyperparameter harms training, it is reverted immediately.

It supports
- PyTorch Adapter or Integration
- Multi-paramter Tuning
- Config-Driven Tuning

## When does a rollback triggers?
A rollback is triggered when training performance regresses significantly and consistently.

At a high level:
- Training or validation metrics may be smoothed over a short window
- The current metric is compared against the best recent value
- A rollback is triggered if:
    1. The relative regression exceeds a configurable threshold 
    2. this condition persists for multiple consecutive tuning steps
- This prevent rollback on single noisy measurements, and expected short-term fluctuations

## What happens during rollback?
When rollback is triggered:
- The most recetn hyperparameter update is rejected
- Hyperparameters are restored to the last known good configuration
- A cooldown period is entered to prevent rapid oscillation
- So rollback restores tuned hyperparameters not the full optimizer or model state.

Restoring only hyperparameters is a deliberate design choice that balances:
1. safety
2. performance
3. framework independence
4. restoring optimize state is expensive and framework-specific
5. most instabilities are caused by unsafe hyperparameters
6. hyperparameter rollback is sufficient to prevent divergence in practice

## Cooldown behavior
After rollback, AutoTuneNet enters a short cooldown window during which:
- further rollbacks are temporarily suppressed
- training is allowed to stabilize
- exploration can resume safely afterward

This avoid repeated rollback loops in noisy regions.

### Configuration
All stability parameters are configurable, including:
1. regression threshold
2. number of consecutive failures
3. cooldown length

## Evalution and Benchmarks:
The current relase focuses on:
1. correctness
2. safety
3. integration quality
4. test coverage

### Formal benchmarks agains:
- fixed hyperparameters
- learning rate schedulers
- offline hyperparameter search
are not yet included. Benchmarking and comparative evaluation are planned, and community contributions in this area are very welcome.

### Config-Driven Tuning
AutoTuneNet supports configuration-driven hyperparameter tuning with built-in safety mechanisms.
AutoTuneNet can be configured entirely via a tuning config:
```yaml
tuning: 
  tune_n_steps: 1
  warmup_epochs: 2
  max_delta: 0.5
```
- Example
```python
from autotunenet.factory import build_pytorch_autotunenet

adapter = build_pytorch_autotunenet(
  torch_optimizer=torch_optimizer,
  raw_config={
    "parameter_space": {
      "lr": [1e-4, 1e-2]
    },
    "tuning": {
      "tune_n_steps": 2,
      "warmup_epochs": 3,
      "max_delta": 0.5
    }
  }
)
```

### Safety-Aware Adaptation
AutoTuneNet performs online hyperparameter adaptation with built-in safety mechanism:
- **warmup phase**: delays tuning until early training stabilizes
- **bounded updates(`max_delta`)**: limits how much a hyperparameter can change per step
- **Rollback & Cooldown**: prevents repeated destabilizing updates

These mechanisms ensure that adaptive tuning remains bounded and predictable, even under non-stationary training dynamics.

### Safety Controls

| Parameter | Description |
|---------|------------|
| `warmup_epochs` | Delays tuning until early training stabilizes |
| `max_delta` | Limits how much a hyperparameter can change per step |
| `tune_n_steps` | Controls tuning frequency |


## Testing
AutoTuneNet is fully unit tested.
```bash
python -m pytest -v
```
Tests cover:
1. optimizer lifecycle
2. stability logic
3. rollback behavior
4. PyTorch adapter
5. config loading

## Folder Structure
```bash
autotunenet/
├── AutoTuneNet/   # Bayesian optimizer, parameter space
├── safeguards/    # Stability and rollback logic
├── adapters/      # Framework integrations (PyTorch)
├── config/        # Config schema & loaders
├── logging/       # Structured logging
├── benchmarks/    # Benchmarks(Fixed_lr, offline_HPO, scheduler, stress_test, autotunenet)
```

# License
MIT License

# Contributing
Contributions are welcome.
- Open issues for bugs or ideas
- PRs fro improvement or adapters
- Tests required for new features

# Acknowledgements
Built on top of:
- Optuna
- PyTorch
Inspired by real-world ML systems where stability matters more than speed.
