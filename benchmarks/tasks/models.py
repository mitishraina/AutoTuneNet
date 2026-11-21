"""
Models used by the benchmark tasks.

Layer construction order is part of each model's definition: with a fixed
seed it decides the initial weights, so changing it changes every result.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTCNN(nn.Module):
    """Small CNN for MNIST (1x28x28): Conv(1->32) -> Conv(32->64) -> FC(128) -> FC(10)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class CIFARCNN(nn.Module):
    """
    CNN for CIFAR-10 (3x32x32): three conv + batch-norm + max-pool blocks
    (64, 128, 256 channels), then FC(1024) -> FC(10), dropout 0.3.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(256 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.dropout(self.pool(F.relu(self.bn1(self.conv1(x)))))
        x = self.dropout(self.pool(F.relu(self.bn2(self.conv2(x)))))
        x = self.dropout(self.pool(F.relu(self.bn3(self.conv3(x)))))
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


class ToyMLP(nn.Module):
    """Tiny MLP for the synthetic toy task (CI and smoke tests)."""

    def __init__(self, in_features: int = 2, num_classes: int = 4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_features, 32), nn.ReLU(), nn.Linear(32, num_classes))

    def forward(self, x):
        return self.net(x)
