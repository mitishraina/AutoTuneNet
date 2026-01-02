import torch
import json
from pathlib import Path

from benchmarks.common.model import CNN
from benchmarks.common.dataset import get_mnist_loaders
from benchmarks.common.train_utils import train_model

def main():
    config = {
        "batch_size": 64,
        "epochs": 10,
        "learning_rate": 0.01, # Fixed learning rate for MNIST dataset
        "seed": 42,
        "optimizer": "Adam"
    }
    
    torch.manual_seed(config["seed"])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, val_loader = get_mnist_loaders(
        batch_size = config["batch_size"],
        seed=config["seed"]
    )
    
    model = CNN().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"]
    )
    
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        epochs=config["epochs"],
    )
    
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
        
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(history, f, indent=2)
        
    print("Fixed LR benchmark completed.")
    print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
    print(f"Final validation accuracy: {history['val_accuracy'][-1]:.4f}")
    
if __name__ == "__main__":
    main()