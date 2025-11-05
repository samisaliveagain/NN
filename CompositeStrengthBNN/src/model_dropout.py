import torch
import torch.nn as nn

class MCDropoutNN(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)
