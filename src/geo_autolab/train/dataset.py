from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from geo_autolab.config import resolve_path
from geo_autolab.geo import latlon_to_unit_xyz

from .config import TrainConfig
from .geocells import (
    HierarchyInfo,
    canonical_country_label,
    canonical_region_label,
    compute_hierarchy_info,
    geo_region_label,
    load_metadata_sidecar,
)
from .transforms import build_transforms


@dataclass(slots=True)
class GeoManifestRecord:
    image_path: str
    latitude: float
    longitude: float
    geocell_id: int
    split: str
    domain: str = "unknown"
    source: str = "unknown"
    group_label: str = "unknown"
    country: str = "unknown"
    region: str = "unknown"
    country_id: int = 0
    region_id: int = 0


def _coerce_record(row: dict[str, str]) -> GeoManifestRecord:
    return GeoManifestRecord(
        image_path=row["image_path"],
        latitude=float(row["latitude"]),
        longitude=float(row["longitude"]),
        geocell_id=int(row["geocell_id"]),
        split=row["split"],
        domain=row.get("domain", "unknown"),
        source=row.get("source", "unknown"),
        group_label=row.get("group_label", "unknown"),
        country=row.get("country", "unknown"),
        region=row.get("region", "unknown"),
        country_id=int(row.get("country_id", 0)),
        region_id=int(row.get("region_id", 0)),
    )


def _resolve_group_label(record: GeoManifestRecord, train_config: TrainConfig) -> str:
    if train_config.group_key == "domain":
        return f"domain:{record.domain}"
    if train_config.group_key == "source":
        return f"source:{record.source}"
    return geo_region_label(
        latitude=record.latitude,
        longitude=record.longitude,
        lat_bins=train_config.geo_region_lat_bins,
        lon_bins=train_config.geo_region_lon_bins,
    )


def load_manifest(
    path: str | Path,
    split: str,
    limit: int | None = None,
    train_config: TrainConfig | None = None,
    hierarchy_info: HierarchyInfo | None = None,
) -> list[GeoManifestRecord]:
    source = resolve_path(path)
    if not source.exists():
        raise FileNotFoundError(f"Manifest not found: {source}")

    metadata_rows = load_metadata_sidecar(source)
    rows: list[GeoManifestRecord] = []

    def _enrich_record(record: GeoManifestRecord) -> GeoManifestRecord:
        metadata = metadata_rows.get(record.image_path, {})
        record.country = canonical_country_label(metadata.get("country"))
        record.region = canonical_region_label(
            metadata.get("country"),
            metadata.get("region"),
            metadata.get("sub_region"),
        )
        if hierarchy_info is not None:
            record.country_id = hierarchy_info.country_to_id.get(record.country, 0)
            record.region_id = hierarchy_info.region_to_id.get(record.region, 0)
        if train_config is not None:
            record.group_label = _resolve_group_label(record, train_config)
        return record

    if source.suffix == ".jsonl":
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                record = _coerce_record({key: str(value) for key, value in payload.items()})
                if record.split == split:
                    rows.append(_enrich_record(record))
    else:
        with source.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                record = _coerce_record(row)
                if record.split == split:
                    rows.append(_enrich_record(record))

    if limit is not None:
        rows = rows[:limit]
    return rows


class GeoDataset(Dataset[dict[str, Tensor | str]]):
    def __init__(self, records: Iterable[GeoManifestRecord], image_size: int, train_config: TrainConfig, training: bool) -> None:
        self.records = list(records)
        self.transform = build_transforms(image_size, train_config.augmentation, training=training)
        self.include_positive_pairs = training and train_config.positive_pair_sampling
        self.positive_pair_fallback = train_config.positive_pair_fallback
        self.index_by_geocell: dict[int, list[int]] = defaultdict(list)
        if self.include_positive_pairs:
            for index, record in enumerate(self.records):
                self.index_by_geocell[record.geocell_id].append(index)

    def __len__(self) -> int:
        return len(self.records)

    def _load_record_tensors(self, record: GeoManifestRecord, prefix: str = "") -> dict[str, Tensor]:
        image = Image.open(resolve_path(record.image_path)).convert("RGB")
        latlon = torch.tensor([record.latitude, record.longitude], dtype=torch.float32)
        return {
            f"{prefix}image": self.transform(image),
            f"{prefix}latlon": latlon,
            f"{prefix}unit_xyz": latlon_to_unit_xyz(latlon),
            f"{prefix}geocell_id": torch.tensor(record.geocell_id, dtype=torch.long),
            f"{prefix}country_id": torch.tensor(record.country_id, dtype=torch.long),
            f"{prefix}region_id": torch.tensor(record.region_id, dtype=torch.long),
        }

    def _sample_positive_index(self, index: int) -> int | None:
        candidates = self.index_by_geocell.get(self.records[index].geocell_id, [])
        if len(candidates) > 1:
            chosen = index
            while chosen == index:
                chosen = candidates[int(torch.randint(0, len(candidates), (1,)).item())]
            return chosen
        if self.positive_pair_fallback == "self":
            return index
        return None

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        record = self.records[index]
        sample: dict[str, Tensor | str] = {
            **self._load_record_tensors(record),
            "domain": record.domain,
            "source": record.source,
            "group_label": record.group_label,
            "country": record.country,
            "region": record.region,
        }
        if self.include_positive_pairs:
            positive_index = self._sample_positive_index(index)
            if positive_index is not None:
                sample.update(self._load_record_tensors(self.records[positive_index], prefix="positive_"))
        return sample


def collate_geo_batch(samples: list[dict[str, Tensor | str]]) -> dict[str, Tensor | list[str]]:
    tensor_keys = ("image", "latlon", "unit_xyz", "geocell_id", "country_id", "region_id")
    batch: dict[str, Tensor | list[str]] = {
        key: torch.stack([sample[key] for sample in samples]) for key in tensor_keys  # type: ignore[arg-type]
    }
    for optional_key in (
        "positive_image",
        "positive_latlon",
        "positive_unit_xyz",
        "positive_geocell_id",
        "positive_country_id",
        "positive_region_id",
    ):
        if optional_key in samples[0]:
            batch[optional_key] = torch.stack([sample[optional_key] for sample in samples])  # type: ignore[arg-type]
    batch["domain"] = [str(sample["domain"]) for sample in samples]
    batch["source"] = [str(sample["source"]) for sample in samples]
    batch["group_label"] = [str(sample["group_label"]) for sample in samples]
    batch["country"] = [str(sample["country"]) for sample in samples]
    batch["region"] = [str(sample["region"]) for sample in samples]
    return batch


def build_dataloaders(
    image_size: int,
    train_config: TrainConfig,
    hierarchy_info: HierarchyInfo | None = None,
) -> tuple[DataLoader, DataLoader]:
    if hierarchy_info is None:
        hierarchy_info = compute_hierarchy_info(train_config.manifest_path)
    train_records = load_manifest(
        train_config.manifest_path,
        split="train",
        limit=train_config.max_train_samples,
        train_config=train_config,
        hierarchy_info=hierarchy_info,
    )
    val_records = load_manifest(
        train_config.manifest_path,
        split="val",
        limit=train_config.max_val_samples,
        train_config=train_config,
        hierarchy_info=hierarchy_info,
    )

    train_dataset = GeoDataset(train_records, image_size=image_size, train_config=train_config, training=True)
    val_dataset = GeoDataset(val_records, image_size=image_size, train_config=train_config, training=False)

    common = {
        "num_workers": train_config.num_workers,
        "pin_memory": train_config.pin_memory,
        "collate_fn": collate_geo_batch,
    }
    if train_config.num_workers > 0 and train_config.prefetch_factor is not None:
        common["prefetch_factor"] = train_config.prefetch_factor
    sampler = None
    if train_config.balance_groups and train_records:
        group_counts = Counter(record.group_label for record in train_records)
        weights = torch.tensor([1.0 / max(1, group_counts[record.group_label]) for record in train_records], dtype=torch.double)
        sampler = WeightedRandomSampler(weights, num_samples=len(train_dataset), replacement=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=len(train_dataset) > train_config.batch_size,
        persistent_workers=train_config.persistent_workers and train_config.num_workers > 0,
        **common,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.eval_batch_size,
        shuffle=False,
        drop_last=False,
        persistent_workers=train_config.persistent_workers and train_config.num_workers > 0,
        **common,
    )
    return train_loader, val_loader
