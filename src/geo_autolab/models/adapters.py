from __future__ import annotations

import torch
from torch import Tensor, nn


class FeatureAdapter(nn.Module):
    def __init__(self, input_dim: int, bottleneck_dim: int, dropout: float, gate_init: float) -> None:
        super().__init__()
        self.down = nn.Linear(input_dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, features: Tensor) -> Tensor:
        residual = features
        adapted = self.down(features)
        adapted = self.activation(adapted)
        adapted = self.dropout(adapted)
        adapted = self.up(adapted)
        mixed = residual + torch.tanh(self.gate) * adapted
        return self.norm(mixed)

