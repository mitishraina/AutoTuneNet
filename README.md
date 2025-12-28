# AutoTuneNet

A generic, open-source Python library that enables **self-optimizing model training** by dynamically tuning hyperparameters during training using **Bayesian Optimization**.

Instead of traditional manually tuning learning rates, batch sizes, or regularization values before training, AutoTuneNet continuously observes training behavior and automatically adjusts hyperparameters to improve convergence and performance.


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
pip install AutoTuneNet
```
## Quick Usage
```bash
import torch 
import torch.nn as nn
import torch.optim as optim

from AutoTuneNet.core.parameters import ParameterSpace
from AutoTuneNet.core.bayesian_optimizer import BayesianOptimizer
from AutoTuneNet.adapters.pytorch.adapter import PyTorchHyperparameterAdapter

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

If a suggested hyperparameter harms training, it is reverted immediately.

It supports
- PyTorch Adapter or Integration
- Multi-paramter Tuning
- Config-Driven Tuning


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