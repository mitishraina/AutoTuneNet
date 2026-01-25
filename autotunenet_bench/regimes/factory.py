from .data_shift import LabelNoiseShift
from .optimizer_noise import GradientNoiseRegime

def build_regime(regime_config: dict):
    if regime_config is None:
        return None
    
    regime_type = regime_config["type"]
    
    if regime_type == "label_noise":
        return LabelNoiseShift(
            start_epoch=regime_config["start_epoch"],
            noise_ratio=regime_config["noise_ratio"],
            seed=regime_config.get("seed", 42)
        )
        
    elif regime_type == "gradient_noise":
        return GradientNoiseRegime(
            start_epoch=regime_config["start_epoch"],
            std=regime_config.get("std", 0.01)
        )
        
    raise ValueError(f"Unknown regime type: {regime_type}")