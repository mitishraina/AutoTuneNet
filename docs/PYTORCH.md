# PyTorch Integration

AutoTuneNet integrates via a thin adapter.

## Supported hooks
- 'on_step_end(metric)'
- 'on_epoch_end(metric)'
- 'on_validation_end(metric)'

## Recommended Usage

Use Validation metrics:
```python
adapter.on_validation_end(metric=val_accuracy)
```

## Supported Optimizers
Works with any optimizer that uses param_groups:
- Adam
- SGD
- RMSProp etc

# Safety

AutoTuneNet is designed to never break training.
## What can go wrong in Tuning?
- exploding loss
- NaNs
- divergence
- catastrophic hyperparameters

## How AutoTuneNet prevents this?
- detects large regressions
- requires consecutive failures(patience)
- enforces cooldown after rollback
- restores last safe configuration

## Logging
All important events are logged:
- parameter updates
- rollbacks
- stabilization events

Logs can be used for debugging and audit trails.