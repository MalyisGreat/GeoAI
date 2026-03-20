from __future__ import annotations

import torch
from torch import Tensor, nn

from .adapters import FeatureAdapter
from .backbone import TimmBackbone
from .config import ModelConfig
from .heads import MultiTaskGeoHead


class GeoLocalizationModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = TimmBackbone(config.backbone)
        if config.adapter.enabled:
            self.adapter = FeatureAdapter(
                input_dim=self.backbone.output_dim,
                bottleneck_dim=config.adapter.bottleneck_dim,
                dropout=config.adapter.dropout,
                gate_init=config.adapter.gate_init,
            )
        else:
            self.adapter = nn.Identity()
        self.head = MultiTaskGeoHead(
            input_dim=self.backbone.output_dim,
            hidden_dim=config.head.hidden_dim,
            embedding_dim=config.head.embedding_dim,
            country_classes=config.head.country_classes,
            region_classes=config.head.region_classes,
            geocell_classes=config.head.geocell_classes,
            dropout=config.head.dropout,
            predict_country=config.head.predict_country,
            predict_region=config.head.predict_region,
            predict_geocell=config.head.predict_geocell,
            predict_uncertainty=config.head.predict_uncertainty,
            decode_topk=config.head.decode_topk,
            max_offset_norm=config.head.max_offset_norm,
            decode_confidence_threshold=config.head.decode_confidence_threshold,
            decode_confidence_sharpness=config.head.decode_confidence_sharpness,
        )

    def set_geocell_centroids(self, centroids: torch.Tensor) -> None:
        self.head.set_geocell_centroids(centroids)

    def forward(self, images: Tensor) -> dict[str, Tensor | None]:
        features = self.backbone(images)
        features = self.adapter(features)
        return self.head(features)
