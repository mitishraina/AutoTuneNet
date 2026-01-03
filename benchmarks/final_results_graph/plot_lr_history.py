import json
import matplotlib.pyplot as plt

def load_lr(path):
    with open(path) as f:
        return json.load(f)
    
def main():
    lr_files = {
        "Scheduler": "benchmarks/scheduler/results/lr_history.json",
        "AutoTuneNet": "benchmarks/autotunenet/results/lr_history.json",
    }
    
    plt.figure(figsize=(8, 5))
    
    for name, path in lr_files.items():
        lr = load_lr(path)
        plt.plot(lr, label=name)
        
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate vs Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    main()