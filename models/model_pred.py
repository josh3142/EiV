import torch
from torch import nn

from models.eiv_layer import EIVDropout

class MLP(nn.Module):
    def __init__(self, in_features: int, p: int = 0.5,
        n_zeta: int = 0, n_zeta_mean: bool = False, **kwargs):
        super().__init__()
        self.n_hidden = 1024
        self.dropout = nn.Dropout(p = p) if (n_zeta == 0 or n_zeta_mean) \
            else EIVDropout(p = p, n_zeta = n_zeta)
        self.mlp = nn.Sequential(
            self.dropout,
            nn.Linear(in_features, self.n_hidden),
            nn.ReLU(),
            self.dropout,
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(),
            self.dropout,
            nn.Linear(self.n_hidden, 10)
        )
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.mlp(x)