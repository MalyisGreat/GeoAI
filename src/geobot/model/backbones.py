from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.skip(x)


class ConvGeoBackbone(nn.Module):
    def __init__(self, variant: str) -> None:
        super().__init__()
        widths = {
            "conv_geo_tiny": [32, 64, 96, 128],
            "conv_geo_large": [64, 128, 256, 384],
        }[variant]
        layers = []
        in_channels = 3
        for width in widths:
            layers.append(ConvBlock(in_channels, width, stride=2))
            in_channels = width
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = widths[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return x


class TimmBackbone(nn.Module):
    def __init__(self, model_name: str, pretrained: bool) -> None:
        super().__init__()
        import timm  # type: ignore

        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        self.out_dim = int(getattr(self.model, "num_features"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_backbone(name: str, fallback_name: str, pretrained: bool) -> nn.Module:
    if name.startswith("timm:"):
        model_name = name.split(":", 1)[1]
        try:
            return TimmBackbone(model_name, pretrained=pretrained)
        except Exception:
            return ConvGeoBackbone(fallback_name)
    return ConvGeoBackbone(name)
