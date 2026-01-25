import subprocess
from pathlib import Path
import json

from autotunenet_bench.metrics.regret import RegretMetric

CONFIG_DIR = Path("autotunenet_bench/configs")
RESULTS_DIR = Path("autotunenet_bench/results")

CONFIGS = {
    "fixed_lr": "mnist_fixed_lr.yaml",
    "scheduler": "mnist_scheduler.yaml",
    "autotunenet": "mnist_autotunenet.yaml",
}


def run_benchmark(name, config_file):
    print(f"\n=== Running {name} ===")
    subprocess.run(
        [
            "python",
            "-m",
            "autotunenet_bench.run_benchmarks",
            "--config",
            str(CONFIG_DIR / config_file),
        ],
        check=True,
    )


def load_val_loss(run_name):
    path = RESULTS_DIR / run_name / "epoch_metrics.json"
    with open(path, "r") as f:
        return json.load(f)["val_loss"]


def main():
    # -----------------------------
    # 1. Run all benchmarks
    # -----------------------------
    for name, cfg in CONFIGS.items():
        run_benchmark(name, cfg)

    # -----------------------------
    # 2. Compute regret vs fixed LR
    # -----------------------------
    baseline_path = RESULTS_DIR / "mnist_fixed_lr" / "epoch_metrics.json"
    regret_metric = RegretMetric(baseline_path)

    summary = {}

    for name in ["scheduler", "autotunenet"]:
        run_path = RESULTS_DIR / f"mnist_{name}" / "epoch_metrics.json"
        regret = regret_metric.compute(run_path)
        summary[name] = regret

    # -----------------------------
    # 3. Save regret summary
    # -----------------------------
    out_path = RESULTS_DIR / "regret_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Regret Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
