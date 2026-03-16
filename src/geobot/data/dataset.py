from __future__ import annotations

import io
import random
import re
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from geobot.utils.geo import tensor_latlon_to_unit
from geobot.utils.io import load_manifest

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CATEGORICAL_AUX_COLUMNS = (
    "country",
    "region",
    "sub-region",
    "city",
    "land_cover",
    "climate",
    "soil",
    "drive_side",
)
NUMERIC_AUX_COLUMNS = ("road_index", "dist_sea")


def safe_column_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()


def _pil_to_tensor(image: Image.Image, image_size: int, augment: bool) -> torch.Tensor:
    image = image.convert("RGB")
    image = ImageOps.fit(image, (image_size, image_size), method=Image.Resampling.BICUBIC)
    if augment:
        if random.random() < 0.2:
            image = ImageOps.mirror(image)
        if random.random() < 0.3:
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
        if random.random() < 0.3:
            image = ImageEnhance.Color(image).enhance(random.uniform(0.85, 1.15))
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array)


class ImageResolver:
    def __init__(self, image_root: str | Path):
        self.image_root = Path(image_root)
        self._index: dict[str, Path] | None = None
        self._archive_index: dict[str, tuple[Path, str]] | None = None
        self._zip_handles: dict[Path, zipfile.ZipFile] = {}

    def _build_index(self) -> None:
        self._index = {}
        if not self.image_root.exists():
            return
        for file_path in self.image_root.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() not in {".zip", ".parquet", ".csv"}:
                self._index[file_path.name] = file_path

    def resolve(self, image_relpath: str) -> Path:
        direct = self.image_root / image_relpath
        if direct.exists():
            return direct
        if self._index is None:
            self._build_index()
        assert self._index is not None
        fallback = self._index.get(Path(image_relpath).name)
        if fallback is None:
            raise FileNotFoundError(f"Image not found under {self.image_root}: {image_relpath}")
        return fallback

    def _build_archive_index(self) -> None:
        self._archive_index = {}
        index_path = self.image_root / "zip_member_index.parquet"
        if not index_path.exists():
            return
        frame = pd.read_parquet(index_path)
        self._archive_index = {
            str(row["filename"]): (Path(str(row["archive_path"])), str(row["member_name"]))
            for _, row in frame.iterrows()
        }

    def load(
        self,
        image_relpath: str,
        *,
        archive_path: str | None = None,
        archive_member_name: str | None = None,
    ) -> Image.Image:
        if archive_path and archive_member_name:
            archive_path_obj = Path(archive_path)
            if archive_path_obj not in self._zip_handles:
                self._zip_handles[archive_path_obj] = zipfile.ZipFile(archive_path_obj, "r")
            with self._zip_handles[archive_path_obj].open(archive_member_name) as handle:
                image_bytes = handle.read()
            return Image.open(io.BytesIO(image_bytes)).convert("RGB").copy()
        try:
            path = self.resolve(image_relpath)
            return Image.open(path).convert("RGB").copy()
        except FileNotFoundError:
            if self._archive_index is None:
                self._build_archive_index()
            assert self._archive_index is not None
            archive_info = self._archive_index.get(Path(image_relpath).name)
            if archive_info is None:
                raise
            archive_path_obj, archive_member_name = archive_info
            if archive_path_obj not in self._zip_handles:
                self._zip_handles[archive_path_obj] = zipfile.ZipFile(archive_path_obj, "r")
            with self._zip_handles[archive_path_obj].open(archive_member_name) as handle:
                image_bytes = handle.read()
            return Image.open(io.BytesIO(image_bytes)).convert("RGB").copy()


@dataclass
class LabelMaps:
    coarse_to_idx: dict[str, int]
    fine_to_idx: dict[str, int]
    aux_categorical: dict[str, dict[str, int]] = field(default_factory=dict)
    aux_numeric_stats: dict[str, tuple[float, float]] = field(default_factory=dict)


def _row_to_item(row: pd.Series, image_tensor: torch.Tensor) -> dict[str, torch.Tensor | str]:
    latlon = torch.tensor(
        [float(row["latitude"]), float(row["longitude"])], dtype=torch.float32
    )
    item: dict[str, torch.Tensor | str] = {
        "image": image_tensor,
        "latlon": latlon,
        "coord_unit": tensor_latlon_to_unit(latlon.unsqueeze(0)).squeeze(0),
        "coarse_idx": torch.tensor(int(row["coarse_idx"]), dtype=torch.long),
        "fine_idx": torch.tensor(int(row["fine_idx"]), dtype=torch.long),
        "image_id": str(row["image_id"]),
    }
    for column, value in row.items():
        if column.startswith("aux_") and column.endswith("_idx"):
            item[column] = torch.tensor(int(value), dtype=torch.long)
        elif column.startswith("aux_") and column.endswith("_value"):
            item[column] = torch.tensor(float(value), dtype=torch.float32)
    return item


class GeoDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        image_root: str | Path,
        image_size: int,
        augment: bool,
    ) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.resolver = ImageResolver(image_root)
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.frame.iloc[index]
        image = _pil_to_tensor(
            self.resolver.load(
                str(row["image_relpath"]),
                archive_path=None if pd.isna(row.get("archive_path")) else str(row.get("archive_path")),
                archive_member_name=None
                if pd.isna(row.get("archive_member_name"))
                else str(row.get("archive_member_name")),
            ),
            self.image_size,
            self.augment,
        )
        return _row_to_item(row, image)


class StreamingGeoDataset(IterableDataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        image_root: str | Path,
        image_size: int,
        augment: bool,
        shuffle_buffer_size: int,
        shuffle_archives: bool,
        seed: int,
        worker_sharding_mode: str = "auto",
    ) -> None:
        super().__init__()
        self.frame = frame.reset_index(drop=True).copy()
        self.image_root = Path(image_root)
        self.image_size = image_size
        self.augment = augment
        self.shuffle_buffer_size = max(1, shuffle_buffer_size)
        self.shuffle_archives = shuffle_archives
        self.seed = seed
        self.worker_sharding_mode = worker_sharding_mode
        self._iteration = 0
        self._archive_groups = self._build_archive_groups()

    def _build_archive_groups(self) -> list[tuple[Path, list[dict[str, object]]]]:
        if "archive_path" not in self.frame.columns or self.frame["archive_path"].isna().all():
            raise ValueError("StreamingGeoDataset requires archive_path in the manifest.")
        grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
        for record in self.frame.to_dict("records"):
            archive_path = record.get("archive_path")
            member_name = record.get("archive_member_name")
            if archive_path is None or pd.isna(archive_path) or member_name is None or pd.isna(member_name):
                continue
            grouped[str(archive_path)].append(record)
        archive_groups: list[tuple[Path, list[dict[str, object]]]] = []
        for archive_path, records in grouped.items():
            records.sort(key=lambda row: str(row["archive_member_name"]))
            archive_groups.append((Path(archive_path), records))
        archive_groups.sort(key=lambda item: str(item[0]))
        return archive_groups

    def __len__(self) -> int:
        return len(self.frame)

    def _iter_archive_records(
        self,
        archive_path: Path,
        records: list[dict[str, object]],
        rng: random.Random,
    ):
        record_lookup = {str(record["archive_member_name"]): record for record in records}
        shuffle_buffer: list[dict[str, torch.Tensor | str]] = []
        with zipfile.ZipFile(archive_path, "r") as archive:
            for info in archive.infolist():
                record = record_lookup.get(info.filename)
                if record is None:
                    continue
                with archive.open(info.filename) as handle:
                    image_bytes = handle.read()
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB").copy()
                image_tensor = _pil_to_tensor(image, self.image_size, self.augment)
                item = _row_to_item(pd.Series(record), image_tensor)
                if self.shuffle_buffer_size <= 1:
                    yield item
                    continue
                shuffle_buffer.append(item)
                if len(shuffle_buffer) >= self.shuffle_buffer_size:
                    yield shuffle_buffer.pop(rng.randrange(len(shuffle_buffer)))
        while shuffle_buffer:
            yield shuffle_buffer.pop(rng.randrange(len(shuffle_buffer)))

    def _iter_record_subset(
        self,
        archive_path: Path,
        records: list[dict[str, object]],
        rng: random.Random,
    ):
        shuffle_buffer: list[dict[str, torch.Tensor | str]] = []
        with zipfile.ZipFile(archive_path, "r") as archive:
            for record in records:
                member_name = str(record["archive_member_name"])
                with archive.open(member_name) as handle:
                    image_bytes = handle.read()
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB").copy()
                image_tensor = _pil_to_tensor(image, self.image_size, self.augment)
                item = _row_to_item(pd.Series(record), image_tensor)
                if self.shuffle_buffer_size <= 1:
                    yield item
                    continue
                shuffle_buffer.append(item)
                if len(shuffle_buffer) >= self.shuffle_buffer_size:
                    yield shuffle_buffer.pop(rng.randrange(len(shuffle_buffer)))
        while shuffle_buffer:
            yield shuffle_buffer.pop(rng.randrange(len(shuffle_buffer)))

    def _resolve_worker_sharding_mode(self, num_workers: int) -> str:
        if self.worker_sharding_mode != "auto":
            return self.worker_sharding_mode
        if num_workers <= 1 or num_workers <= len(self._archive_groups):
            return "archive"
        return "record"

    @staticmethod
    def _slice_records_for_worker(
        records: list[dict[str, object]],
        worker_id: int,
        num_workers: int,
    ) -> list[dict[str, object]]:
        if num_workers <= 1:
            return records
        chunk_size = max(1, (len(records) + num_workers - 1) // num_workers)
        start = worker_id * chunk_size
        end = min(len(records), start + chunk_size)
        return records[start:end]

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        self._iteration += 1
        rng = random.Random(self.seed + self._iteration + worker_id * 9973)
        archive_groups = list(self._archive_groups)
        if self.shuffle_archives:
            rng.shuffle(archive_groups)
        sharding_mode = self._resolve_worker_sharding_mode(num_workers)
        if sharding_mode == "record":
            for archive_path, records in archive_groups:
                worker_records = self._slice_records_for_worker(records, worker_id, num_workers)
                if worker_records:
                    yield from self._iter_record_subset(archive_path, worker_records, rng)
            return

        worker_archives = archive_groups[worker_id::num_workers]
        for archive_path, records in worker_archives:
            yield from self._iter_archive_records(archive_path, records, rng)


def load_frame(path: str | Path) -> pd.DataFrame:
    return load_manifest(path)


def split_train_val(
    frame: pd.DataFrame, *, val_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(frame) < 2:
        return frame.copy(), frame.copy()
    shuffled = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_size = max(1, min(len(frame) - 1, int(round(len(frame) * val_fraction))))
    val = shuffled.iloc[:val_size].reset_index(drop=True)
    train = shuffled.iloc[val_size:].reset_index(drop=True)
    return train, val


def attach_label_indices(
    train_frame: pd.DataFrame, val_frame: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, LabelMaps]:
    coarse_vocab = sorted(train_frame["coarse_cell"].astype(str).unique().tolist())
    fine_vocab = sorted(train_frame["fine_cell"].astype(str).unique().tolist())
    coarse_to_idx = {label: idx for idx, label in enumerate(coarse_vocab)}
    fine_to_idx = {label: idx for idx, label in enumerate(fine_vocab)}

    categorical_maps: dict[str, dict[str, int]] = {}
    numeric_stats: dict[str, tuple[float, float]] = {}

    for column in CATEGORICAL_AUX_COLUMNS:
        if column not in train_frame.columns:
            continue
        safe = safe_column_name(column)
        train_values = train_frame[column].fillna("UNK").astype(str)
        val_values = val_frame[column].fillna("UNK").astype(str)
        vocab = sorted(train_values.unique().tolist())
        mapping = {label: idx for idx, label in enumerate(vocab)}
        categorical_maps[safe] = mapping
        train_frame = train_frame.copy()
        val_frame = val_frame.copy()
        train_frame[f"aux_{safe}_idx"] = train_values.map(mapping).fillna(0).astype(int)
        val_frame[f"aux_{safe}_idx"] = val_values.map(mapping).fillna(0).astype(int)

    for column in NUMERIC_AUX_COLUMNS:
        if column not in train_frame.columns:
            continue
        safe = safe_column_name(column)
        train_numeric = pd.to_numeric(train_frame[column], errors="coerce")
        val_numeric = pd.to_numeric(val_frame[column], errors="coerce")
        mean = float(train_numeric.mean()) if train_numeric.notna().any() else 0.0
        std = float(train_numeric.std()) if train_numeric.notna().any() else 1.0
        std = std if std > 1e-6 else 1.0
        numeric_stats[safe] = (mean, std)
        train_frame = train_frame.copy()
        val_frame = val_frame.copy()
        train_frame[f"aux_{safe}_value"] = ((train_numeric.fillna(mean) - mean) / std).astype(float)
        val_frame[f"aux_{safe}_value"] = ((val_numeric.fillna(mean) - mean) / std).astype(float)

    def _map_labels(frame: pd.DataFrame) -> pd.DataFrame:
        mapped = frame.copy()
        mapped["coarse_idx"] = mapped["coarse_cell"].map(coarse_to_idx).fillna(0).astype(int)
        mapped["fine_idx"] = mapped["fine_cell"].map(fine_to_idx).fillna(0).astype(int)
        return mapped

    return _map_labels(train_frame), _map_labels(val_frame), LabelMaps(
        coarse_to_idx=coarse_to_idx,
        fine_to_idx=fine_to_idx,
        aux_categorical=categorical_maps,
        aux_numeric_stats=numeric_stats,
    )
