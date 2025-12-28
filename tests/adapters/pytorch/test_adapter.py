import torch
import torch.nn as nn
import torch.optim as optim

from src.AutoTuneNet.parameters import ParameterSpace
from src.AutoTuneNet.bayesian_optimizer import BayesianOptimizer
from src.adapters.pytorch.adapter import PyTorchHyperParameterAdapter

def test_adapter_updates_learning_rate():
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
    
    adapter.step(metric=1.0) # fake metric
    
    updated_lr = optimizer.param_groups[0]['lr']
    
    assert initial_lr != updated_lr
    
    
def test_adapter_multiple_steps_safe():
    model = nn.Linear(2, 1)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    space = ParameterSpace({"lr": (0.001, 0.1)})
    autotune = BayesianOptimizer(space, seed=123)
    
    adapter = PyTorchHyperParameterAdapter(
        torch_optimizer=optimizer,
        autotune_optimizer=autotune,
        tune_n_steps=1
    )
    
    for _ in range(10):
        adapter.step(metric=-0.01)
        
        
def test_adapter_respects_tuning_frequency():
    model = nn.Linear(2, 1)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    space = ParameterSpace({"lr": (0.001, 0.1)})
    autotune = BayesianOptimizer(space, seed=42)
    
    adapter = PyTorchHyperParameterAdapter(
        torch_optimizer=optimizer,
        autotune_optimizer=autotune,
        tune_n_steps=2
    )
    
    lr_1 = optimizer.param_groups[0]['lr']
    adapter.step(metric=-0.1)
    lr_2 = optimizer.param_groups[0]['lr']
    
    assert lr_1 == lr_2  # No change on first step or no tuning yet
    
    adapter.step(metric=1.0)
    lr_3 = optimizer.param_groups[0]['lr']
    
    assert lr_2 != lr_3  # Change should happen on second step or tuning should happen