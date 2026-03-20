from __future__ import annotations

import json
from pathlib import Path

import torch

from geo_autolab.utils import ensure_dir

from .config import ModelConfig
from .model import GeoLocalizationModel


def export_bundle(output_dir: str | Path, model: GeoLocalizationModel, config: ModelConfig) -> Path:
    target_dir = ensure_dir(output_dir)
    state_path = target_dir / "model.pt"
    metadata_path = target_dir / "metadata.json"
    torch.save({"state_dict": model.state_dict(), "config": config.model_dump()}, state_path)
    metadata = {
        "image_size": config.image_size,
        "dynamic_batch": config.export.dynamic_batch,
        "outputs": ["embedding", "unit_xyz", "geocell_logits", "uncertainty"],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return state_path

