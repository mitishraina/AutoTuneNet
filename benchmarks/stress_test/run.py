import torch
import json
from pathlib import Path

from benchmarks.common.model import CNN
from benchmarks.common.dataset import get_mnist_loaders
from benchmarks.common.train_utils import train_model

from autotunenet.parameters import ParameterSpace
from autotunenet.bayesian_optimizer import BayesianOptimizer
from autotunenet.adapters.pytorch.adapter import PyTorchHyperParameterAdapter

def run_fixed_lr(model, train_loader, val_loader, device, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    return train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        epochs=epochs
    )
    
def run_scheduler(model, train_loader, val_loader, device, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs
    )
    
    def on_epoch_end(epoch, val_loss, val_accuracy):
        scheduler.step()
        
    return train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        on_epoch_end=on_epoch_end
    )
    
def run_autotunenet(model, train_loader, val_loader, device, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    param_space = ParameterSpace({
        "lr": (1e-4, 1e-1)
    })
    
    autotune_optimizer = BayesianOptimizer(
        param_space=param_space,
        smoothing_window=3
    )
    
    adapter = PyTorchHyperParameterAdapter(
        torch_optimizer=optimizer,
        autotune_optimizer=autotune_optimizer,
        tune_n_steps=1
    )
    
    def on_epoch_end(epoch, val_loss, val_accuracy):
        adapter.on_epoch_end(metric=-val_loss)
        
    return train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        on_epoch_end=on_epoch_end
    )
    
    
def main():
    config = {
        "batch_size": 64,
        "epochs": 10,
        "initial_lr": 0.1,
        "seed": 42,
        "stress_test": "bad_initial_lr"
    }
    
    torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, val_loader = get_mnist_loaders(
        batch_size=config["batch_size"],
        seed=config["seed"]
    )
    
    results = {}
    
    # Fixed LR
    model = CNN().to(device)
    results["fixed_lr"] = run_fixed_lr(
        model, train_loader, val_loader, device, config["epochs"]
    )
    
    #Scheduler
    model = CNN().to(device)
    results["scheduler"] = run_scheduler(
        model, train_loader, val_loader, device, config["epochs"]
    )
    
    # AutoTuneNet
    model = CNN().to(device)
    results["autotunenet"] = run_autotunenet(
        model, train_loader, val_loader, device, config["epochs"]
    )
    
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
        
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    
    print("Stress Test Benchmark completed.")
    for key, history in results.items():
        print(
            f"{key}: final val_loss={history['val_loss'][-1]:.4f}"
            f"val_accuracy={history['val_accuracy'][-1]:.4f}"
        )
        
if __name__ == "__main__":
    main()