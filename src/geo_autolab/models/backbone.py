from __future__ import annotations

import torch
import timm
from torch import Tensor, nn

from .config import BackboneConfig


class TimmBackbone(nn.Module):
    def __init__(self, config: BackboneConfig) -> None:
        super().__init__()
        self.config = config
        self.model = timm.create_model(
            config.timm_name,
            pretrained=config.pretrained,
            num_classes=0,
            drop_path_rate=config.drop_path_rate,
        )
        if config.checkpoint_gradients and hasattr(self.model, "set_grad_checkpointing"):
            self.model.set_grad_checkpointing(True)
        if not config.train_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        self.output_dim = self._infer_output_dim()

    def _infer_output_dim(self) -> int:
        input_size = getattr(self.model, "default_cfg", {}).get("input_size", (3, 224, 224))
        if len(input_size) != 3:
            input_size = (3, 224, 224)
        sample = torch.zeros((1, int(input_size[0]), int(input_size[1]), int(input_size[2])), dtype=torch.float32)
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            features = self.model(sample)
        if was_training:
            self.model.train()
        if features.ndim != 2:
            raise ValueError(f"Expected pooled embedding from timm backbone, got shape {tuple(features.shape)}")
        return int(features.shape[-1])

    def forward(self, images: Tensor) -> Tensor:
        features = self.model(images)
        if features.ndim != 2:
            raise ValueError(f"Expected pooled embedding from timm backbone, got shape {tuple(features.shape)}")
        return features
