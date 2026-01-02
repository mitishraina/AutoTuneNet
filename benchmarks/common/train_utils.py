import torch
import torch.nn.functional as F

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            loss = F.cross_entropy(output, target)
            total_loss += loss.item()
            
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
            
    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy

def train_model(model, train_loader, val_loader, optimizer, device, epochs: int, on_epoch_end=None):
    """
    Generic training loop.
    on_epoch_end:
        Optional callable(epoch, val_loss, val_accuracy)
    """
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
    }
    
    for epoch in range(epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, device
        )
        
        val_loss, val_accuracy = evaluate(
            model, val_loader, device
        )
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        
        if on_epoch_end is not None:
            on_epoch_end(epoch, val_loss, val_accuracy)
            
    return history