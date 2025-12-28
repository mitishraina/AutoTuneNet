import torch
import torch.nn as nn
import torch.optim as optim

from autotunenet.parameters import ParameterSpace
from autotunenet.bayesian_optimizer import BayesianOptimizer
from autotunenet.adapters.pytorch.adapter import PyTorchHyperParameterAdapter

model = nn.Linear(10, 1)

optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4
)

param_space = ParameterSpace({
    "lr": (1e-4, 1e-1),
    "momentum": (0.0, 0.99),
    "weight_decay": (1e-6, 1e-2)
})

autotune = BayesianOptimizer(param_space, seed=42)

adapter = PyTorchHyperParameterAdapter(
    torch_optimizer=optimizer,
    autotune_optimizer=autotune
)

for epoch in range(20):
    val_metric = -abs(optimizer.param_groups[0]["lr"] - 0.01)
    adapter.step(metric=val_metric)
    
    print(
        f"Epoch {epoch} | "
        f"lr={optimizer.param_groups[0]['lr']:.6f} | "
        f"momentum={optimizer.param_groups[0]['momentum']:.4f} | "
        f"weight_decay={optimizer.param_groups[0]['weight_decay']:.6f}"
    )