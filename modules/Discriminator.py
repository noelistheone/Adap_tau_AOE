import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(input_dim, input_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
