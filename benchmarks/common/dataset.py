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

def get_cifar_loaders(
    batch_size: int = 128,
    val_split: float = 0.1,
    seed: int = 42,
):
    """
    Returns train and validation DataLoaders for CIFAR-10
    """
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    ])


    full_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=None
    )

    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    from torch.utils.data import Subset
    
    class ApplyTransform(torch.utils.data.Dataset):
        def __init__(self, dataset, transform):
            self.dataset = dataset
            self.transform = transform
            
        def __getitem__(self, idx):
            x, y = self.dataset[idx]
            if self.transform:
                x = self.transform(x)
            return x, y
            
        def __len__(self):
            return len(self.dataset)

    generator = torch.Generator().manual_seed(seed)
    train_ds_raw, val_ds_raw = random_split(
        full_dataset,
        [train_size, val_size],
        generator=generator
    )

    train_ds = ApplyTransform(train_ds_raw, train_transform)
    val_ds = ApplyTransform(val_ds_raw, val_transform)

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
