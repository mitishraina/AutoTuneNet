import torch
import torch.nn as nn
import torch.optim as optim

from src.core.parameters import ParameterSpace
from src.core.bayesian_optimizer import BayesianOptimizer
from src.adapters.pytorch.adapter import PyTorchHyperParameterAdapter

def test_epoch_hook_updates_params():
    model = nn.Linear(2, 1)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    space = ParameterSpace({"lr": (0.001, 0.1)})
    autotune = BayesianOptimizer(space, seed=42)
    
    adapter = PyTorchHyperParameterAdapter(
        torch_optimizer = optimizer,
        autotune_optimizer = autotune, 
    )
    
    lr_before = optimizer.param_groups[0]['lr']
    adapter.on_epoch_end(metric=-0.1)
    lr_after = optimizer.param_groups[0]['lr']
    
    assert lr_before != lr_after
    
    
def test_hook_respects_frequency():
    model = nn.Linear(2, 1)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    space = ParameterSpace({"lr": (0.001, 0.1)})
    autotune = BayesianOptimizer(space, seed=42)
    
    adapter = PyTorchHyperParameterAdapter(
        optimizer,
        autotune,
        tune_n_steps=2
    )    
    
    lr_1 = optimizer.param_groups[0]['lr']
    adapter.on_epoch_end(metric=-0.1)
    lr_2 = optimizer.param_groups[0]['lr']
    
    assert lr_1 == lr_2
    
    adapter.on_epoch_end(metric=-0.1)
    lr_3 = optimizer.param_groups[0]['lr']
    assert lr_3 != lr_2