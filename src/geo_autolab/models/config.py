from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class BackboneConfig(BaseModel):
    timm_name: str = "convnext_tiny"
    pretrained: bool = True
    drop_path_rate: float = 0.1
    train_backbone: bool = True
    checkpoint_gradients: bool = False


class AdapterConfig(BaseModel):
    enabled: bool = True
    bottleneck_dim: int = 256
    dropout: float = 0.1
    gate_init: float = 0.25


class HeadConfig(BaseModel):
    country_classes: int = 0
    region_classes: int = 0
    geocell_classes: int = 4096
    hidden_dim: int = 512
    embedding_dim: int = 256
    dropout: float = 0.2
    predict_country: bool = True
    predict_region: bool = True
    predict_geocell: bool = True
    predict_uncertainty: bool = True
    decode_topk: int = 8
    max_offset_norm: float = 0.08
    decode_confidence_threshold: float = 0.08
    decode_confidence_sharpness: float = 18.0


class LossConfig(BaseModel):
    country_weight: float = 0.4
    region_weight: float = 0.4
    geocell_weight: float = 1.0
    geodesic_weight: float = 0.75
    embedding_weight: float = 0.02
    offset_weight: float = 0.15
    hierarchy_consistency_weight: float = 0.05
    label_smoothing: float = 0.05
    contrastive_temperature: float = 0.07
    spatial_geocell_radius_km: float = 250.0
    spatial_geocell_topk: int = 16


class ExportConfig(BaseModel):
    example_image_size: int = 224
    dynamic_batch: bool = True


class ModelConfig(BaseModel):
    image_size: int = 224
    backbone: BackboneConfig = Field(default_factory=BackboneConfig)
    adapter: AdapterConfig = Field(default_factory=AdapterConfig)
    head: HeadConfig = Field(default_factory=HeadConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    backbone_family: Literal["convnext", "efficientnet", "vit", "other"] = "convnext"

    @classmethod
    def local_default(cls) -> "ModelConfig":
        return cls(
            image_size=224,
            backbone=BackboneConfig(timm_name="convnext_tiny", pretrained=True, drop_path_rate=0.1),
            adapter=AdapterConfig(enabled=True, bottleneck_dim=256, dropout=0.1, gate_init=0.25),
            head=HeadConfig(geocell_classes=4096, hidden_dim=512, embedding_dim=256, dropout=0.2),
        )
