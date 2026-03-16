from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .backbones import build_backbone


class GeoLocator(nn.Module):
    def __init__(
        self,
        *,
        backbone_name: str,
        fallback_backbone: str,
        pretrained: bool,
        hidden_dim: int,
        embedding_dim: int,
        dropout: float,
        num_coarse_classes: int,
        num_fine_classes: int,
    ) -> None:
        super().__init__()
        self.backbone = build_backbone(backbone_name, fallback_backbone, pretrained)
        self.projection = nn.Sequential(
            nn.Linear(self.backbone.out_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.coarse_head = nn.Linear(embedding_dim, num_coarse_classes)
        self.fine_head = nn.Linear(embedding_dim, num_fine_classes)
        self.coord_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(images)
        embedding = F.normalize(self.projection(features), dim=-1)
        coarse_logits = self.coarse_head(embedding)
        fine_logits = self.fine_head(embedding)
        coord_unit = F.normalize(self.coord_head(embedding), dim=-1)
        return {
            "embedding": embedding,
            "coarse_logits": coarse_logits,
            "fine_logits": fine_logits,
            "coord_unit": coord_unit,
        }
