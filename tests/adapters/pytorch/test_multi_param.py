import torch 
import torch.nn as nn
import torch.optim as optim

from src.AutoTuneNet.parameters import ParameterSpace
from src.AutoTuneNet.bayesian_optimizer import BayesianOptimizer   
from src.adapters.pytorch.adapter import PyTorchHyperParameterAdapter

def test_multi_param_update():
    model = nn.Linear(2, 1)
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9
    )
    
    space = ParameterSpace({
        "lr": (0.001, 0.1),
        "momentum": (0.8, 0.99)
    })
    
    autotune = BayesianOptimizer(space, seed=42)
    adapter = PyTorchHyperParameterAdapter(optimizer, autotune)
    
    initial = optimizer.param_groups[0].copy()
    adapter.step(metric=-0.01)
    
    updated = optimizer.param_groups[0]
    
    assert (
        updated["lr"] != initial["lr"]
        or updated["momentum"] != initial["momentum"]
    )