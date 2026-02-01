import torch
import torch.nn as nn
import torch.nn.functional as F

# mnist
# class CNN(nn.Module):
#     """
#     Simple CNN for MNIST benchmarks
#     Architecture:
#         - Conv(1 -> 32)
#         - Conv(32 -> 64)
#         - FC(128)
#         - FC(10)
#     """
    
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
#         self.fc1 = nn.Linear(64 * 7 * 7, 128)
#         self.fc2 = nn.Linear(128, 10)
        
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.max_pool2d(x, 2)
        
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, 2)
        
#         x = torch.flatten(x, 1)
        
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
        
#         return x


#cifar10
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        # self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        # self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        # self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        # self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        # self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        # self.dropout = nn.Dropout(0.2)
        self.dropout = nn.Dropout(0.3)
        self._to_linear = None
        # self._infer_fc()
        self.eval()
        self._infer_fc()
        self.train()

        # self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc1 = nn.Linear(self._to_linear, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def _infer_fc(self):
        with torch.no_grad():
            x = torch.zeros(1, 3, 32, 32) 
            # x = self.pool(F.relu(self.bn1(self.conv1(x))))
            # x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
            self._to_linear = x.numel()

    def forward(self, x):
        # x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # x = self.dropout(x)
        # x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # x = self.dropout(x)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)