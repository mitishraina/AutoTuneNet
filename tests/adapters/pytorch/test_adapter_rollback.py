import torch
import torch.nn as nn
import torch.optim as optim

from autotunenet.parameters import ParameterSpace
from autotunenet.bayesian_optimizer import BayesianOptimizer
from autotunenet.adapters.pytorch.adapter import PyTorchHyperParameterAdapter

def test_adapter_rollback_on_exception():
    model = nn.Linear(2, 1)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    space = ParameterSpace({"lr": (0.001, 0.1)})
    autotune = BayesianOptimizer(space, seed=42)
    
    adapter = PyTorchHyperParameterAdapter(
        torch_optimizer=optimizer,
        autotune_optimizer=autotune,
        tune_n_steps=1
    )
    
    initial_lr = optimizer.param_groups[0]['lr']
    
    adapter.step(metric=-0.0001)
    lr_after_good = optimizer.param_groups[0]['lr']
    
    adapter.step(metric=-100.0)
    lr_after_bad = optimizer.param_groups[0]['lr']
    
    assert lr_after_bad == lr_after_good