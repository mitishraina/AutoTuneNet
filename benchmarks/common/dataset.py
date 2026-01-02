import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_mnist_loaders(
    batch_size: int = 64,
    val_split: float = 0.1,
    seed: int = 42,
):
    """
    Returns train and validation DataLoaders for MNIST
    """
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    full_train = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    val_size = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size
    
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(
        full_train,
        [train_size, val_size],
        generator=generator
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader