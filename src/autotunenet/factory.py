from autotunenet.parameters import ParameterSpace
from autotunenet.bayesian_optimizer import BayesianOptimizer
from autotunenet.adapters.pytorch.adapter import PyTorchHyperParameterAdapter
from autotunenet.config.schema import AutoTuneConfig

def build_pytorch_autotunenet(
    torch_optimizer,
    raw_config: dict,
):
    """
    Factory that builds AutoTuneNet components for PyTorch from a validated configuration.
    """
    
    config = AutoTuneConfig(raw_config)
    
    param_space = ParameterSpace(config.parameter_space)
    
    autotune_optimizer = BayesianOptimizer(
        param_space=param_space,
        smoothing_window=config.metrics.get("smoothing_window", 3)
    )
    
    tuning_cfg = config.tuning
    
    tune_n_steps = tuning_cfg.get("tune_n_steps", 1)
    warmup_epochs = tuning_cfg.get("warmup_epochs", 0)
    warmup_metric_threshold = tuning_cfg.get(
        "warmup_metric_threshold"
    )
    max_delta = tuning_cfg.get("max_delta")
    
    
    adapter = PyTorchHyperParameterAdapter(
        torch_optimizer=torch_optimizer,
        autotune_optimizer=autotune_optimizer,
        tune_n_steps=tune_n_steps,
        warmup_epochs=warmup_epochs,
        warmup_metric_threshold=warmup_metric_threshold,
        max_delta=max_delta,
    )
    
    return adapter


# How users will use the factory:
# adapter = build_pytorch_autotunenet(
#     torch_optimizer,
#     raw_config={
#         "parameter_space": {
#             "lr": [1e-4, 1e-2]
#         },
#         "tuning": {
#             "warmup_epochs": 2,
#             "max_delta": 0.5
#         }
#     }
# )