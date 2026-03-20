from __future__ import annotations

import torch
from torch import nn

from .backbone import BackboneBundle
from .heads import GeoHead


class GeoLocator(nn.Module):
    def __init__(self, backbone: BackboneBundle, head: GeoHead) -> None:
        super().__init__()
        self.encoder = backbone.encoder
        self.head = head
        self.feature_dim = backbone.feature_dim
        self.backbone_source = backbone.source

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor | None]:
        features = self.encoder(inputs)
        return self.head(features)
