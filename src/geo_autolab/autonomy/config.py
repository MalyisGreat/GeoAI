from __future__ import annotations

from pydantic import BaseModel, Field

from geo_autolab.eval.config import EvalConfig
from geo_autolab.train.config import TrainConfig


class AutoLabConfig(BaseModel):
    name: str = "local-autolab"
    run_root: str = "runs/autonomy"
    history_path: str = "runs/autonomy/history.jsonl"
    max_cycles: int = 3
    candidates_per_cycle: int = 3
    continue_until_target: bool = False
    target_median_km: float | None = None
    target_within_100km: float | None = None
    target_geocell_top1: float | None = None
    model_config_path: str = "configs/model/local_baseline.yaml"
    train: TrainConfig = Field(default_factory=TrainConfig.local_default)
    evaluation: EvalConfig = Field(default_factory=EvalConfig)
