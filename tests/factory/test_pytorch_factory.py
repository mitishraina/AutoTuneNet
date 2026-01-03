import pytest
from unittest.mock import Mock

from autotunenet.factory import build_pytorch_autotunenet
from autotunenet.adapters.pytorch.adapter import PyTorchHyperParameterAdapter

def test_factory_builds_adapter_with_config():
    torch_optimizer = Mock()
    torch_optimizer.param_groups = [{'lr': 0.01}]
    
    raw_config={
        "parameter_space": {
            "lr": [1e-4, 1e-2]
        },
        "tuning": {
            "tune_n_steps": 2,
            "warmup_epochs": 3,
            "max_delta": 0.5
        }
    }
    
    adapter = build_pytorch_autotunenet(
        torch_optimizer=torch_optimizer,
        raw_config=raw_config
    )
    
    assert isinstance(adapter, PyTorchHyperParameterAdapter)
    
    assert adapter.tune_n_steps == 2
    assert adapter.warmup_epochs == 3
    assert adapter.max_delta == 0.5
    
    assert adapter.torch_optimizer is torch_optimizer
    assert adapter.autotune_optimizer is not None
    
def test_factory_uses_safe_defaults():
    torch_optimizer = Mock()
    torch_optimizer.param_groups = [{'lr': 0.01}]
    
    raw_config={
        "parameter_space": {
            "lr": [1e-4, 1e-2]
        }
    }
    
    adapter = build_pytorch_autotunenet(
        torch_optimizer=torch_optimizer,
        raw_config=raw_config
    )
    
    assert adapter.tune_n_steps == 1
    assert adapter.warmup_epochs == 0
    assert adapter.max_delta is None