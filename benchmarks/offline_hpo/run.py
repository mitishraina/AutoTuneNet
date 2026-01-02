import torch
import json
from pathlib import Path
import optuna

from benchmarks.common.model import CNN
from benchmarks.common.dataset import get_mnist_loaders
from benchmarks.common.train_utils import train_model

def objective(trial, device, train_loader, val_loader):
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    
    model = CNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        epochs=3
    )
    
    return history["val_loss"][-1]

def main():
    config = {
        "batch_size": 64,
        "epochs": 10,
        "trial_epochs": 3,
        "n_trials": 8,
        "seed": 42,
        "optimizer": "Adam",
        "search_space": {
            "lr": [1e-4, 1e-1] 
        }
    }
    
    torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, val_loader = get_mnist_loaders(
        batch_size=config["batch_size"],
        seed=config["seed"]
    )
    
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, device, train_loader, val_loader),
        n_trials=config["n_trials"]
    )
    
    best_lr = study.best_params["lr"]
    
    model = CNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)
    
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        epochs=config["epochs"]
    )
    
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
        
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(history, f, indent=2)
        
    with open(output_dir / "best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
        
    print("Offline HPO benchmark completed.")
    print(f"Best LR: {best_lr:.6f}")
    print(f"Final val_loss: {history['val_loss'][-1]:.4f}")
    print(f"Final val_accuracy: {history['val_accuracy'][-1]:.4f}")
    
if __name__ == "__main__":
    main()