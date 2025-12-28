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

### Refer **quickstart.md** for quick guide on usage.