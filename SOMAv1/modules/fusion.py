# modules/fusion.py
import torch
import torch.nn as nn

class GatedFusion(nn.Module):
    """
    Simple gated fusion: out = g * x1 + (1 - g) * x2
    where g = sigmoid(Linear([x1, x2])).
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.gate = nn.Linear(2 * embed_dim, embed_dim)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate(torch.cat([x1, x2], dim=-1)))
        return g * x1 + (1.0 - g) * x2
