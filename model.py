import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_channels , hidden_size, out_channels):
        super().__init__()
        self.flatten = nn.Flatten()
        self.ReLU    = nn.ReLU(inplace=True)
        self.linear = nn.Sequential(
            nn.Linear(in_channels, hidden_size//8),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 8, hidden_size // 8),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 8, hidden_size // 8),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 8, hidden_size // 8),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 8, hidden_size // 8),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 8, hidden_size // 8),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size//8, hidden_size//4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 4, hidden_size // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 4, hidden_size // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 4, hidden_size // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 4, hidden_size // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 4, hidden_size // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size// 4, hidden_size//2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size//2, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 4, hidden_size // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 4, hidden_size // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 4, hidden_size // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 4, hidden_size // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 4, hidden_size // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 4, hidden_size // 8),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 8, hidden_size // 8),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 8, hidden_size // 8),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 8, hidden_size // 8),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 8, hidden_size // 8),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 8, hidden_size // 8),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 8, out_channels),
        )




    def forward(self, x):
        x_ = self.flatten(x)
        x_ = self.ReLU(x_)
        predicion = self.linear(x_)
        return predicion

