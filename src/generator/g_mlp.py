import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim, y_dim, hidden_dim, output_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + y_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()   
        )

    def forward(self, x):
        return self.net(x)
