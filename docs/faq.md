# FAQ

## Using AutoTuneNet

### Does it replace Optuna?

No, it uses Optuna. Optuna's TPE sampler proposes the configurations through its ask–tell interface. AutoTuneNet adds what TPE does not provide inside a live run: credit assignment to the configuration that actually ran, bounded updates, stability checks, rollback and warmup. If you can afford many full training runs, use Optuna directly (offline HPO). The benchmark includes that comparison.

### Is it AutoML or architecture search?

No. It tunes numeric optimizer settings (`lr`, `weight_decay`, `momentum`, …) during training. Changing the architecture mid-run is out of scope.

### What metric should I pass?

A **higher-is-better** score measured on validation data with the configuration that just ran: `-val_loss`, or `val_accuracy`. A training loss can be used with `on_step_end`, but it is noisy and rewards overfitting. Don't pass test-set metrics.

### How often should I call it?

Once per epoch is the default. For long epochs, call `on_validation_end` every N steps with a fixed validation subset ([PyTorch guide](pytorch.md#tuning-more-often-than-once-per-epoch)). Each call is one observation for TPE, so short runs need either more calls or a small `n_startup_trials`.

### Why are the first proposals random?

TPE samples `n_startup_trials` configurations at random before it fits its model (default 10, Optuna's default). With one call per epoch, a 10-epoch run never gets past the random phase, so set it to 2–3.

### Why did the learning rate only change by 50%?

`max_delta` clips every proposal to `current × (1 ± max_delta)`. This keeps changes smooth. Use a larger `max_delta` or `None` to allow bigger jumps.

### Does a rollback restore my model?

No, only the hyperparameters. If you also want the weights back, check `adapter.last_rollback` and reload your own checkpoint ([example](pytorch.md#rollback-scope)).

### Does it work with a learning-rate scheduler?

Not on the same setting. Both would write `lr`. Either let AutoTuneNet own `lr`, or keep the scheduler and tune other settings ([details](pytorch.md#learning-rate-schedulers)).

### What does it cost?

One TPE ask/tell per tuning call, about 1 ms with a few hundred observations. It never trains extra models.

### Is it deterministic?

Yes, given a `seed` for `BayesianOptimizer` and the same sequence of metrics. Nondeterminism in your training (cuDNN, data loading order) produces different metrics and therefore different decisions.

## Benchmarks

### Why is CIFAR-10 accuracy only about 70–75%?

The task is a small CNN trained for 10 epochs with no augmentation, the same for every method. It measures how methods compare under identical, cheap conditions, not state-of-the-art accuracy. A ResNet with augmentation and longer training would reach 90%+ for every method.

### Why does the cosine schedule beat AutoTuneNet in some cells?

For a known, fixed number of epochs on stationary data, a decaying schedule is close to optimal and costs nothing to "learn". AutoTuneNet has to discover the decay from about 8 observations. Its intended advantage is in runs where the right schedule is not known in advance: shifts mid-run and bad starting values. The results report both cases, including the ones it loses.

### Why is PBT not bit-reproducible?

Ray runs the population concurrently. Which members have reported at a synchronisation point depends on scheduling, which can change which member gets copied. Every other method is reproducible for a given seed and device type.

### Why are offline methods n/a for the bad initial learning rate?

They choose their own learning rate per trial, so a forced bad starting value does not apply to them.

### Can I benchmark my own model and dataset?

Yes, with a plugin file and no changes to the package: [Extending the benchmark](benchmarks/extending.md).

### Is my Kaggle token safe?

Commands only check that a token file exists; they never read or print it. Notebooks run under your account and contain no credentials. Runtime state that includes your username is written to a git-ignored `.runtime/` folder.
