from __future__ import annotations

from pydantic import BaseModel, Field


class EvalConfig(BaseModel):
    distance_thresholds_km: list[int] = Field(default_factory=lambda: [1, 25, 100, 750, 2500])
    max_primary_metric_km: float = 2500.0
    min_within_100km: float = 0.08
    min_worst_group_within_100km: float = 0.03
    max_group_gap_ratio: float = 4.0
    max_ece: float = 0.35
    max_confidence_gap: float = 0.35
    suspicious_geocell_gap: float = 0.25

