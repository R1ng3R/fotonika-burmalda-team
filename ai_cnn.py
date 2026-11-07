import torch
import torch.nn as nn


class DoeNet(nn.Module):
    """
    Вход : (B, 4, 128, 128) — 4 исходных изображения
    Выход: (B, 1, 128, 128) — фаза φ ∈ [-π, π]
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),  nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1,  kernel_size=3, padding=1),
        )

    def forward(self, x):
        return torch.tanh(self.net(x)) * torch.pi
