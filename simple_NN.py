import torch
import torch.nn as nn
import torch.nn.functional as F

# A simple normal NN WITHOUT dropout
class NormalNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(1, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 1)

    def forward(self, x):
        # Layer 1: Linear + Activation
        x = F.relu(self.fc1(x))
        # Layer 2: Linear + Activation
        x = F.relu(self.fc2(x))
        # Layer 3: Final Linear (Output)
        return self.fc3(x)