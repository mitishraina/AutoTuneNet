import json
from pathlib import Path

class RegretMetric:
    """
    Expirical cumulative regret against a baseline controller.
    """
    def __init__(self, baseline_metrics_path: Path):
        self.baseline_metrics_path = baseline_metrics_path
        
        with open(baseline_metrics_path, "r") as f:
            data = json.load(f)
            self.baseline = [row["val_loss"] for row in data]
            
    
    def compute(self, controller_metrics_path: Path):
        with open(controller_metrics_path, "r") as f:
            data = json.load(f)
            controller = [row["val_loss"] for row in data]
            
        T = min(len(self.baseline), len(controller))
        
        regret = 0.0
        per_epoch = []
        
        for t in range(T):
            delta = controller[t] - self.baseline[t]
            regret += delta
            per_epoch.append(delta)
            
        return {
            "cumulative_regret": regret,
            "mean_regret": regret / T if T > 0 else 0.0,
            "per_epoch_regret": per_epoch,
            "horizon": T
        }
        