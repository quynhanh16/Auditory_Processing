import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    def __init__(self, d_plus_1: int, m: int):
        super(CNNModel, self).__init__()

        # Input: (batch_size, 1, d+1, m)
        self.conv1 = nn.Conv2d(1, 12, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(12, 16, kernel_size=3, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=3)
        self.dropout = nn.Dropout(0.25)

        # Calculate the flattened size after conv + pool
        def get_flattened_size():
            with torch.no_grad():
                dummy = torch.zeros(1, 1, d_plus_1, m)
                x = self.pool(self.conv2(self.conv1(dummy)))
                return x.view(1, -1).shape[1]

        self.flat_dim = get_flattened_size()

        self.fc1 = nn.Linear(self.flat_dim, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # (B, 12, H-2, W-2)
        x = F.relu(self.conv2(x))  # (B, 16, H-4, W-4)
        x = self.pool(x)  # (B, 16, (H-4)//3, (W-4)//3)
        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
