import torch
import json
from pathlib import Path

from benchmarks.common.model import CNN
from benchmarks.common.dataset import get_mnist_loaders
from benchmarks.common.train_utils import train_model

from autotunenet.parameters import ParameterSpace
from autotunenet.bayesian_optimizer import BayesianOptimizer
from autotunenet.adapters.pytorch.adapter import PyTorchHyperParameterAdapter

def main():
    config = {
        "batch_size": 64,
        "epochs": 10,
        "initial_lr": 0.01,
        "seed": 42,
        "optimizer": "Adam",
        "tuned_params": ["lr"]
    }
    
    torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, val_loader = get_mnist_loaders(
        batch_size=config["batch_size"],
        seed=config["seed"]
    )
    
    model = CNN().to(device)
    
    torch_optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["initial_lr"]
    )
    
    param_space = ParameterSpace({
        "lr": (1e-4, 1e-1)
    })
    
    autotune_optimizer = BayesianOptimizer(
        param_space=param_space,
        smoothing_window=3
    )
    
    adapter = PyTorchHyperParameterAdapter(
        torch_optimizer=torch_optimizer,
        autotune_optimizer=autotune_optimizer,
        tune_n_steps = 1
    )
    
    lr_history = []
    
    def on_epoch_end(epoch, val_loss, val_accuracy):
        adapter.on_epoch_end(metric=-val_loss) #maximize negative loss
        lr_history.append(torch_optimizer.param_groups[0]['lr'])
        
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=torch_optimizer,
        device=device,
        epochs=config["epochs"],
        on_epoch_end=on_epoch_end
    )
    
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
        
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(history, f, indent=2)
        
    with open(output_dir / "lr_history.json", "w") as f:
        json.dump(lr_history, f, indent=2)
        
    
    print("AutoTuneNet benchmark completed.")
    print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
    print(f"Final validation accuracy: {history['val_accuracy'][-1]:.4f}")
    print(f"Final learning rate: {lr_history[-1]:.6f}")
    
if __name__ == "__main__":
    main()
    