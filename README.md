# AutoTuneNet

A generic, open-source Python library that enables **self-optimizing model training** by dynamically tuning hyperparameters during training using **Bayesian Optimization**.

Instead of manually tuning learning rates, batch sizes, or regularization values before training, AutoTuneNet continuously observes training behavior and automatically adjusts hyperparameters to improve convergence and performance.

---

## Why This Project Exists

Hyperparameter tuning is one of the most time-consuming and error-prone parts of machine learning workflows.

Common problems:
- Manual trial-and-error
- Grid/random search waste compute
- Hyperparameters are fixed before training
- Optimal values often change during training

**AutoTuneNet solves this by making hyperparameter tuning part of the training loop itself.**

---

## Core Idea

AutoTuneNet treats hyperparameter tuning as a **learning problem**.

During training:
1. The model trains normally
2. Training and validation metrics are observed
3. A Bayesian optimizer models the relationship between hyperparameters and performance
4. Hyperparameters are updated **incrementally and safely**
5. Training continues with improved settings

This creates a closed-loop, self-optimizing training system.

---

## What This Is (and Is Not)

### This project is
- A **generic hyperparameter optimization engine**
- **Model-agnostic**
- **Dataset-agnostic**
- Designed to plug into existing training loops
- Suitable for research and production workflows

### This project is nto
- A single ML model
- Offline AutoML that runs many full trials
- Grid or random search
- Neural Architecture Search (for now)

---

## Design Philosophy

- **Framework-agnostic core**  
  The Bayesian optimization logic does not depend on PyTorch or TensorFlow.

- **Thin framework adapters**  
  Framework-specific code lives in adapters (PyTorch first).

- **Safety first**  
  Guardrails prevent unstable updates and allow rollback.

- **Minimal user code changes**  
  Users should be able to integrate this with a few lines of code.

---

## Quick Example (PyTorch)

```python
from autotunenet import AutoTuneNet

tuner = AutoTuneNet(
    param_space={
        "learning_rate": (1e-5, 1e-2),
        "dropout": (0.1, 0.5)
    }
)

trainer.fit(
    model,
    train_loader,
    val_loader,
    tuner=tuner
)
