from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AugmentationConfig(BaseModel):
    resize_scale_min: float = 0.7
    color_jitter: float = 0.2
    blur_prob: float = 0.15
    grayscale_prob: float = 0.05
    random_erasing_prob: float = 0.15


class TrainConfig(BaseModel):
    manifest_path: str = "data/manifest.csv"
    run_dir: str = "runs/manual"
    init_checkpoint: str | None = None
    batch_size: int = 12
    eval_batch_size: int = 24
    grad_accum_steps: int = 2
    num_workers: int = 4
    max_epochs: int = 5
    learning_rate: float = 3.0e-4
    backbone_lr_scale: float = 0.2
    weight_decay: float = 0.05
    max_grad_norm: float = 1.0
    warmup_fraction: float = 0.1
    amp: bool = True
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int | None = None
    cudnn_benchmark: bool = False
    channels_last: bool = False
    seed: int = 1337
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    eval_every_images: int | None = None
    log_every: int = 20
    positive_pair_sampling: bool = True
    positive_pair_fallback: Literal["self", "none"] = "self"
    balance_groups: bool = True
    group_key: Literal["geo_region", "domain", "source"] = "geo_region"
    geo_region_lat_bins: int = 6
    geo_region_lon_bins: int = 12
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)

    @classmethod
    def local_default(cls) -> "TrainConfig":
        return cls()
