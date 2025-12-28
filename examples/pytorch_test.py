import torch
import torch.nn as nn
import torch.optim as optim

from src.AutoTuneNet.parameters import ParameterSpace
from src.AutoTuneNet.bayesian_optimizer import BayesianOptimizer
from src.adapters.pytorch.adapter import PyTorchHyperParameterAdapter

model = nn.Linear(10, 1)
optimizer = optim.Adam(model.parameters(), lr=0.01)

param_space = ParameterSpace({
    "lr": (1e-4, 1e-1)
})

autotune = BayesianOptimizer(param_space, seed=42)

adapter = PyTorchHyperParameterAdapter(
    torch_optimizer=optimizer,
    autotune_optimizer=autotune,
    tune_n_steps=1
)

for epoch in range(20):
    val_loss = abs(0.01 - optimizer.param_groups[0]["lr"])
    adapter.step(metric=-val_loss)
    
    print(
        f"Epoch {epoch+1} |"
        f"lr={optimizer.param_groups[0]['lr']:.6f} |"
    )