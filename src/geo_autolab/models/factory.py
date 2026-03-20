from __future__ import annotations

from .config import ModelConfig
from .losses import GeoCriterion
from .model import GeoLocalizationModel


def build_model_stack(config: ModelConfig) -> tuple[GeoLocalizationModel, GeoCriterion]:
    model = GeoLocalizationModel(config)
    criterion = GeoCriterion(config.loss)
    return model, criterion

