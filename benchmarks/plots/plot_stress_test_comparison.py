import json
import matplotlib.pyplot as plt

def load_metrics(path):
    with open(path) as f:
        return json.load(f)

def main():
    experiments = {
        "Fixed LR": "benchmarks/stress_test/results/results.json",
        "Scheduler": "benchmarks/stress_test/results/results.json",
        "AutoTuneNet": "benchmarks/stress_test/results/results.json",
    }

    plt.figure(figsize=(8, 5))

    for name, path in experiments.items():
        metrics = load_metrics(path)
        
        if name == "Fixed LR":
            val_loss = metrics["fixed_lr"]["val_loss"]
        elif name == "Scheduler":
            val_loss = metrics["scheduler"]["val_loss"]
        elif name == "AutoTuneNet":
            val_loss = metrics["autotunenet"]["val_loss"]

        plt.plot(val_loss, label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Stress Test: Validation Loss vs Epoch(Bad Initial LR)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
