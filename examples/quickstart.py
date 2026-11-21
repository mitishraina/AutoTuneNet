"""Minimal AutoTuneNet loop: tune the learning rate of a linear model, once per epoch.

    python examples/quickstart.py
"""
import torch
from torch import nn

from autotunenet.adapters.pytorch.adapter import PyTorchHyperParameterAdapter
from autotunenet.bayesian_optimizer import BayesianOptimizer
from autotunenet.parameters import ParameterSpace

torch.manual_seed(0)

# Toy data: y = 3x + noise. Replace with your own DataLoaders.
x = torch.randn(1024, 1)
y = 3 * x + 0.1 * torch.randn(1024, 1)
x_train, y_train, x_val, y_val = x[:800], y[:800], x[800:], y[800:]

model = nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

space = ParameterSpace({"lr": {"low": 1e-4, "high": 1e-1, "log": True}})
tuner = BayesianOptimizer(space, seed=0, n_startup_trials=3)
adapter = PyTorchHyperParameterAdapter(
    torch_optimizer=optimizer,
    autotune_optimizer=tuner,
    warmup_epochs=2,      # leave the first 2 epochs alone
    max_delta=0.5,        # change lr by at most ±50% per epoch
)

for epoch in range(15):
    model.train()
    for i in range(0, len(x_train), 32):
        optimizer.zero_grad()
        loss_fn(model(x_train[i:i + 32]), y_train[i:i + 32]).backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_loss = loss_fn(model(x_val), y_val).item()

    adapter.on_epoch_end(metric=-val_loss)          # higher is better
    print(f"epoch {epoch:2d}  val_loss={val_loss:.4f}  lr={optimizer.param_groups[0]['lr']:.5f}"
          f"{'  rollback' if adapter.last_rollback else ''}")
