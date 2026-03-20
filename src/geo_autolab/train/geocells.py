from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from geo_autolab.config import resolve_path
from geo_autolab.geo import latlon_to_unit_xyz


@dataclass(slots=True)
class HierarchyInfo:
    country_labels: list[str]
    region_labels: list[str]
    country_to_id: dict[str, int]
    region_to_id: dict[str, int]
    geocell_to_country: Tensor
    geocell_to_region: Tensor


def geo_region_label(latitude: float, longitude: float, lat_bins: int, lon_bins: int) -> str:
    lat = min(89.9999, max(-89.9999, latitude))
    lon = ((longitude + 180.0) % 360.0) - 180.0
    lat_idx = min(lat_bins - 1, max(0, int(((lat + 90.0) / 180.0) * lat_bins)))
    lon_idx = min(lon_bins - 1, max(0, int(((lon + 180.0) / 360.0) * lon_bins)))
    return f"geo_region:{lat_idx:02d}-{lon_idx:02d}"


def canonical_country_label(country: str | None) -> str:
    value = (country or "").strip()
    return value if value else "unknown"


def canonical_region_label(country: str | None, region: str | None, sub_region: str | None = None) -> str:
    country_label = canonical_country_label(country)
    region_value = (region or "").strip() or (sub_region or "").strip() or "unknown"
    return f"{country_label}::{region_value}"


def infer_geocell_classes(manifest_path: str | Path) -> int:
    source = resolve_path(manifest_path)
    max_geocell = -1
    if source.suffix == ".jsonl":
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                max_geocell = max(max_geocell, int(payload["geocell_id"]))
    else:
        with source.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                max_geocell = max(max_geocell, int(row["geocell_id"]))
    return max_geocell + 1


def metadata_sidecar_path(manifest_path: str | Path) -> Path:
    source = resolve_path(manifest_path)
    return source.with_name("metadata.jsonl")


def load_metadata_sidecar(manifest_path: str | Path) -> dict[str, dict[str, str]]:
    target = metadata_sidecar_path(manifest_path)
    if not target.exists():
        return {}

    rows: dict[str, dict[str, str]] = {}
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            image_path = str(payload.get("image_path", "")).strip()
            if not image_path:
                continue
            rows[image_path] = {
                "country": str(payload.get("country", "") or ""),
                "region": str(payload.get("region", "") or ""),
                "sub_region": str(payload.get("sub_region", "") or ""),
                "city": str(payload.get("city", "") or ""),
            }
    return rows


def compute_geocell_centroids(manifest_path: str | Path, geocell_classes: int | None = None) -> Tensor:
    source = resolve_path(manifest_path)
    if geocell_classes is None:
        geocell_classes = infer_geocell_classes(source)

    sums = torch.zeros((geocell_classes, 3), dtype=torch.float32)
    counts = torch.zeros((geocell_classes,), dtype=torch.float32)

    def _accumulate(latitude: float, longitude: float, geocell_id: int) -> None:
        xyz = latlon_to_unit_xyz(torch.tensor([latitude, longitude], dtype=torch.float32))
        sums[geocell_id] += xyz
        counts[geocell_id] += 1.0

    if source.suffix == ".jsonl":
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                _accumulate(float(payload["latitude"]), float(payload["longitude"]), int(payload["geocell_id"]))
    else:
        with source.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                _accumulate(float(row["latitude"]), float(row["longitude"]), int(row["geocell_id"]))

    global_fallback = sums.sum(dim=0)
    if torch.linalg.vector_norm(global_fallback).item() == 0.0:
        global_fallback = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    global_fallback = torch.nn.functional.normalize(global_fallback, dim=0)

    for geocell_id in range(geocell_classes):
        if counts[geocell_id].item() == 0.0:
            sums[geocell_id] = global_fallback

    return torch.nn.functional.normalize(sums, dim=-1)


def compute_hierarchy_info(manifest_path: str | Path, geocell_classes: int | None = None) -> HierarchyInfo:
    source = resolve_path(manifest_path)
    if geocell_classes is None:
        geocell_classes = infer_geocell_classes(source)
    metadata_rows = load_metadata_sidecar(source)

    geocell_country_counts: dict[int, Counter[str]] = defaultdict(Counter)
    geocell_region_counts: dict[int, Counter[str]] = defaultdict(Counter)
    country_counts: Counter[str] = Counter()
    region_counts: Counter[str] = Counter()

    with source.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            geocell_id = int(row["geocell_id"])
            metadata = metadata_rows.get(str(row["image_path"]), {})
            country_label = canonical_country_label(metadata.get("country"))
            region_label = canonical_region_label(
                metadata.get("country"),
                metadata.get("region"),
                metadata.get("sub_region"),
            )
            geocell_country_counts[geocell_id][country_label] += 1
            geocell_region_counts[geocell_id][region_label] += 1
            country_counts[country_label] += 1
            region_counts[region_label] += 1

    def _sorted_labels(counter: Counter[str]) -> list[str]:
        labels = [label for label, _count in sorted(counter.items(), key=lambda item: (-item[1], item[0])) if label != "unknown"]
        return ["unknown", *labels]

    country_labels = _sorted_labels(country_counts)
    region_labels = _sorted_labels(region_counts)
    country_to_id = {label: index for index, label in enumerate(country_labels)}
    region_to_id = {label: index for index, label in enumerate(region_labels)}

    geocell_to_country = torch.zeros((geocell_classes,), dtype=torch.long)
    geocell_to_region = torch.zeros((geocell_classes,), dtype=torch.long)

    for geocell_id in range(geocell_classes):
        country_label = "unknown"
        if geocell_id in geocell_country_counts:
            country_label = geocell_country_counts[geocell_id].most_common(1)[0][0]
        region_label = "unknown"
        if geocell_id in geocell_region_counts:
            region_label = geocell_region_counts[geocell_id].most_common(1)[0][0]
        geocell_to_country[geocell_id] = country_to_id.get(country_label, 0)
        geocell_to_region[geocell_id] = region_to_id.get(region_label, 0)

    return HierarchyInfo(
        country_labels=country_labels,
        region_labels=region_labels,
        country_to_id=country_to_id,
        region_to_id=region_to_id,
        geocell_to_country=geocell_to_country,
        geocell_to_region=geocell_to_region,
    )
