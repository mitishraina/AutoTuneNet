from unittest.mock import MagicMock
from autotunenet.adapters.pytorch.adapter import PyTorchHyperParameterAdapter

def test_epoch_warmup_blocks_early_tuning():
    torch_optimizer = MagicMock()
    torch_optimizer.param_groups = [{"lr": 0.01}]
    
    autotune_optimizer = MagicMock()
    autotune_optimizer.suggest.return_value = {"lr": 0.02}
    autotune_optimizer.best_score.return_value = None
    autotune_optimizer.observe.return_value = True
    
    adapter = PyTorchHyperParameterAdapter(
        torch_optimizer=torch_optimizer,
        autotune_optimizer=autotune_optimizer,
        tune_n_steps=1,
        warmup_epochs=2
    )
    
    # First two warmup epochs should not trigger tuning
    adapter.on_epoch_end(metric=-1.0)
    adapter.on_epoch_end(metric=-0.8)
    
    autotune_optimizer.suggest.assert_not_called()
    
    # Third epoch should trigger tuning or warmup complete
    adapter.on_epoch_end(metric=-0.5)
    
    autotune_optimizer.suggest.assert_called_once()
    
def test_metric_threshold_blocks_tuning():
    torch_optimizer = MagicMock()
    torch_optimizer.param_groups = [{"lr": 0.01}]
    
    autotune_optimizer = MagicMock()
    autotune_optimizer.suggest.return_value = {"lr": 0.02}
    autotune_optimizer.best_score.return_value = None
    autotune_optimizer.observe.return_value = True
    
    adapter = PyTorchHyperParameterAdapter(
        torch_optimizer=torch_optimizer,
        autotune_optimizer=autotune_optimizer,
        warmup_metric_threshold=0.3
    )
    
    adapter.on_epoch_end(metric=-1.0)
    adapter.on_epoch_end(metric=-0.6)
    
    autotune_optimizer.suggest.assert_not_called()
    
    adapter.on_epoch_end(metric=-0.2)
    autotune_optimizer.suggest.assert_called_once()


def test_max_delta_clamps_lr():
    torch_optimizer = MagicMock()
    torch_optimizer.param_groups = [{"lr": 0.01}]
    
    autotune_optimizer = MagicMock()
    autotune_optimizer.suggest.return_value = {"lr": 0.1}
    autotune_optimizer.observe.return_value = True
    autotune_optimizer.best_score.return_value = None 
    
    adapter = PyTorchHyperParameterAdapter(
        torch_optimizer=torch_optimizer,
        autotune_optimizer=autotune_optimizer,
        max_delta=0.5
    )
    
    adapter.on_epoch_end(metric=-0.1)
    
    assert torch_optimizer.param_groups[0]["lr"] == 0.015