"""
Dataset loaders for the benchmarks.

Datasets are not stored in the repository. The first time a dataset is
needed it is downloaded into DATA_DIR (<repo>/data, or $AUTOTUNENET_DATA_DIR),
with a progress bar showing percentage, size, speed and time left. To fetch
everything up front, run:

    uv run python -m benchmarks.download_data
"""
import os
from pathlib import Path

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("AUTOTUNENET_DATA_DIR", REPO_ROOT / "data"))

# torchvision dataset class -> (display name, download size, source)
DATASET_INFO = {
    datasets.MNIST: ("MNIST", "about 11 MB", "the PyTorch MNIST mirror"),
    datasets.CIFAR10: ("CIFAR-10", "about 170 MB", "www.cs.toronto.edu (this server can be slow)"),
}


def load_dataset(dataset_cls, train: bool, transform=None):
    """
    Load a torchvision dataset from DATA_DIR, downloading it first if missing.

    The download prints what is being fetched and shows torchvision's progress
    bar; later runs find the files and skip it.
    """
    try:
        return dataset_cls(root=str(DATA_DIR), train=train, download=False, transform=transform)
    except RuntimeError:
        name, size, source = DATASET_INFO.get(dataset_cls, (dataset_cls.__name__, "unknown size", "its source"))
        split = "training" if train else "test"
        print(
            f"[data] {name} ({split} split) not found in {DATA_DIR}.\n"
            f"[data] Downloading {name} ({size}) from {source}. This happens once.\n"
            f"[data] Set AUTOTUNENET_DATA_DIR to keep datasets somewhere else.",
            flush=True,
        )
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        dataset = dataset_cls(root=str(DATA_DIR), train=train, download=True, transform=transform)
        print(f"[data] {name} ready in {DATA_DIR}", flush=True)
        return dataset

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
    
    full_train = load_dataset(datasets.MNIST, train=True, transform=transform)
    
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


    full_dataset = load_dataset(datasets.CIFAR10, train=True, transform=None)

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


def get_mnist_test_loader(batch_size: int = 64):
    """Held-out MNIST test split (10k images), never used for tuning."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_ds = load_dataset(datasets.MNIST, train=False, transform=transform)
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False)


def get_cifar_test_loader(batch_size: int = 128):
    """Held-out CIFAR-10 test split (10k images), never used for tuning."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    ])
    test_ds = load_dataset(datasets.CIFAR10, train=False, transform=transform)
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False)
