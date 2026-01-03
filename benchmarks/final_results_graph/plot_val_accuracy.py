import json
import matplotlib.pyplot as plt

def load_metrics(path):
    with open(path) as f:
        return json.load(f)
    
def main():
    experiments = {
        "Fixed LR": "benchmarks/fixed_lr/results/metrics.json",
        "Scheduler": "benchmarks/scheduler/results/metrics.json",
        "AutoTuneNet": "benchmarks/autotunenet/results/metrics.json",
    }
    
    plt.figure(figsize=(8, 5))
    
    for name, path in experiments.items():
        metrics = load_metrics(path)
        plt.plot(metrics["val_accuracy"], label=name)
        
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy vs Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    main()