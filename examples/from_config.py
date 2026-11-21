"""Build the adapter from a YAML file instead of Python objects.

    python examples/from_config.py
"""
from pathlib import Path

import torch
from torch import nn

from autotunenet.adapters.pytorch.adapter import PyTorchHyperParameterAdapter

model = nn.Linear(10, 1)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
adapter = PyTorchHyperParameterAdapter.from_config(
    optimizer, str(Path(__file__).with_name("autotune.yaml")), seed=0)

x, y = torch.randn(512, 10), torch.randn(512, 1)
for epoch in range(8):
    optimizer.zero_grad()
    loss = nn.functional.mse_loss(model(x), y)
    loss.backward()
    optimizer.step()
    adapter.on_epoch_end(metric=-loss.item())       # use a validation loss in real code
    print(f"epoch {epoch}  loss={loss.item():.4f}  lr={optimizer.param_groups[0]['lr']:.5f}")
