# PyTorch guide

How the adapter interacts with a PyTorch training loop: which hook to call and when, param groups, learning-rate schedulers and checkpoints.

## Hooks

| Method | Call it | Typical metric |
|---|---|---|
| `on_epoch_end(metric)` | after validating at the end of an epoch | `-val_loss` |
| `on_validation_end(metric)` | after any validation pass (e.g. every N steps) | `-val_loss` |
| `on_step_end(metric)` | after an optimizer step | `-train_loss` of a smoothed window (noisy) |
| `step(metric)` | any loop that is not epoch-based | any higher-is-better score |

All four run the same logic (see [How it works](concepts.md#one-tuning-call)). Choose the one that names your call site. `warmup_epochs` and `tune_n_steps` count **calls**, whichever hook you use.

The metric must have been measured with the hyperparameters in effect since the previous call: it is credited to them. Validate first, then call the hook.

```python
for epoch in range(epochs):
    train_one_epoch(model, optimizer)
    val_loss = evaluate(model)                 # measured with this epoch's lr
    adapter.on_epoch_end(metric=-val_loss)     # credits it, then sets next epoch's lr
```

Pass a loss as `-loss`, or a higher-is-better metric (accuracy, F1) as it is. Floats, NumPy scalars and one-element tensors are accepted; `loss.item()` is not required.

### Tuning more often than once per epoch

For long epochs, validate on a held-out subset every few hundred steps and call `on_validation_end`:

```python
for step, batch in enumerate(loader):
    train_step(batch)
    if step % 500 == 499:
        adapter.on_validation_end(metric=-quick_val_loss(model))
```

More calls give TPE more observations per run. Keep the validation subset fixed, so scores are comparable between calls.

## Optimizers

Any `torch.optim.Optimizer` works: AutoTuneNet only reads and writes `optimizer.param_groups`. You can tune any numeric setting the optimizer has:

| Optimizer | Tunable settings (examples) |
|---|---|
| `SGD` | `lr`, `momentum`, `weight_decay`, `dampening` |
| `Adam`, `AdamW` | `lr`, `weight_decay`, `eps` |
| `RMSprop` | `lr`, `alpha`, `momentum`, `weight_decay` |

A name the optimizer does not have, such as `momentum` with Adam, raises `ValueError` at construction and lists the available settings. Tuple-valued settings such as Adam's `betas` cannot be tuned.

## Param groups

Values refer to the first group. Other groups keep their ratio to it:

```python
optimizer = torch.optim.SGD([
    {"params": backbone.parameters(), "lr": 1e-4},
    {"params": head.parameters(),     "lr": 1e-3},
], momentum=0.9)
# The controller sets lr = 2e-4 on the backbone -> head gets 2e-3.
```

Every tuned name must exist in every group. Runnable: [`examples/multi_param.py`](https://github.com/mitishraina/AutoTuneNet/blob/main/examples/multi_param.py).

## Learning-rate schedulers

Don't tune `lr` and run a scheduler on it at the same time: both write `lr`, and the adapter's credit assignment assumes it set the value that ran. Choose one of these:

- Tune `lr` with AutoTuneNet and drop the scheduler. This is what the benchmarks do.
- Keep the scheduler for `lr` and tune other settings (`weight_decay`, `momentum`).

Warmup schedules are covered by `warmup_epochs`: AutoTuneNet leaves the optimizer untouched until warmup ends, so a scheduler can own the first epochs, as long as it stops writing `lr` before tuning starts.

## Mixed precision, gradient clipping, DDP

- **AMP / `GradScaler`:** unaffected. The scaler wraps `optimizer.step()`; AutoTuneNet changes `param_groups` between steps.
- **Gradient clipping:** unaffected.
- **DistributedDataParallel:** every rank must apply the same hyperparameters. Compute the validation metric on all ranks (all-reduce it), seed the `BayesianOptimizer` identically on every rank, and call the hook on every rank. The ranks then make identical decisions.

## Checkpoints

The adapter and the TPE study are ordinary Python objects. To resume a run with the controller's state, pickle the adapter together with your checkpoint:

```python
import pickle
torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, "ckpt.pt")
with open("adapter.pkl", "wb") as f:
    pickle.dump(adapter, f)
```

When restoring, load the optimizer state first, then unpickle the adapter and point it at the restored optimizer: `adapter.torch_optimizer = optimizer`. Without the pickle, a resumed run starts a fresh study (warmup and TPE history start over), which is safe but loses what was learned.

## Rollback scope

A rollback restores **hyperparameters only**. Model weights and optimizer moments are left as they are. If you want weight rollback too, check `adapter.last_rollback` after each call and restore your own last checkpoint when it is `True`:

```python
adapter.on_epoch_end(metric=-val_loss)
if adapter.last_rollback:
    model.load_state_dict(best_state)          # your own copy, kept when val_loss improved
```
