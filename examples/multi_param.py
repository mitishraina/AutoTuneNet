"""Tune learning rate, momentum and weight decay together, with two param groups.

    python examples/multi_param.py
"""
import torch
from torch import nn

from autotunenet.adapters.pytorch.adapter import PyTorchHyperParameterAdapter
from autotunenet.bayesian_optimizer import BayesianOptimizer
from autotunenet.parameters import ParameterSpace
from autotunenet.safeguards.stability import StabilityMonitor

torch.manual_seed(0)
x = torch.randn(2048, 20)
y = (x[:, :5].sum(dim=1) > 0).long()
x_train, y_train, x_val, y_val = x[:1600], y[:1600], x[1600:], y[1600:]

body = nn.Sequential(nn.Linear(20, 64), nn.ReLU())
head = nn.Linear(64, 2)
model = nn.Sequential(body, head)

# Two groups: the body learns 10x slower than the head. AutoTuneNet keeps that ratio.
optimizer = torch.optim.SGD([
    {"params": body.parameters(), "lr": 0.005},
    {"params": head.parameters(), "lr": 0.05},
], momentum=0.9, weight_decay=1e-4)

space = ParameterSpace({
    "lr": {"low": 1e-3, "high": 0.3, "log": True},           # log scale
    "momentum": (0.8, 0.99),                                 # tuple: linear range
    "weight_decay": {"low": 1e-6, "high": 1e-3, "log": True},
})
tuner = BayesianOptimizer(space, seed=0, n_startup_trials=3)
tuner.guard = StabilityMonitor(max_regression=0.1, patience=2, cooldown=2)
adapter = PyTorchHyperParameterAdapter(optimizer, tuner, warmup_epochs=1, max_delta=0.5)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(12):
    model.train()
    for i in range(0, len(x_train), 64):
        optimizer.zero_grad()
        loss_fn(model(x_train[i:i + 64]), y_train[i:i + 64]).backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(x_val)
        val_loss = loss_fn(logits, y_val).item()
        val_acc = (logits.argmax(1) == y_val).float().mean().item()

    adapter.on_epoch_end(metric=-val_loss)
    g0, g1 = optimizer.param_groups
    print(f"epoch {epoch:2d}  val_loss={val_loss:.4f}  acc={val_acc:.3f}  "
          f"lr={g0['lr']:.5f}/{g1['lr']:.5f}  momentum={g0['momentum']:.3f}  wd={g0['weight_decay']:.2e}")
