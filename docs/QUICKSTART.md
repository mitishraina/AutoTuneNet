# QuickStart

This guide shows how to use AutoTuneNet


# Installation(local)
```bash
pip install .
```

### Example
```bash
from autotunenet.parameters import ParameterSpace
from autotunenet.bayesian_optimizer import BayesianOptimizer
from autotunenet.adapters.pytorch.adapter import PyTorchHyperparameterAdapter

optmizer = torch.optim.Adam(model.parameters(), lr=0.01)
param_space = ParameterSpace({
    "lr": (1e-4, 1e-1)
})

autotune = BayesianOptimizer(param_space)
adapter = PyTorchHyperparameterAdapter(
    torch_optimizer = optimizer,
    autotune_optimizer = autotune
)

for epoch in range(epochs):
    train(...)
    val_metric = validate(...)
    adapter.on_epoch_end(metric=val_metric)
```

That's it, AutoTuneNet will
- explore learning rate
- keep the best one
- rollback if training becomes unstable

# Core Concepts

1. Training-time optimization
AutoTuneNet does not run separate trials. It adapts hyperparameters while training is ongoing.

2. Suggest -> Observe loop
- suggest hyperparameters
- apply them
- observe metric
- accept or rollback

This loop repeats safely thorughout training.

3. Bayesian Optmization
AutoTuneNet uses Bayesian Optimization to:
- model the relationship between hyperparameters and performance
- prioritize promising regions
- still explore when uncertain

4. Safety first
Every update is guarded by:
- metric smoothing
- regression detection
- cooldown periods
- rollback to last known good state

Training stability is never compromise.

