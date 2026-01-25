import json
from pathlib import Path

class BenchmarkLogger:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.records = []
        
    def log_epoch(self, record: dict):
        self.records.append(record)
        
    def write_epoch_logs(self):
        assert hasattr(self, "records"), "BenchmarkLogger.records missing"
        with open(self.output_dir / "epoch_metric.json", "w") as f:
            json.dump(self.records, f, indent=2)
            
    def write_stability_summary(self, stability_summary: dict):
        with open(self.output_dir / "stability_metrics.json", "w") as f:
            json.dump(stability_summary, f, indent=2)
            
    def write_compute_metrics(self, metrics: dict):
        with open(self.output_dir / "compute_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
            