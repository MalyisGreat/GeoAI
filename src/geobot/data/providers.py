from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from geobot.utils.geo import assign_grid_cells
from geobot.utils.io import (
    build_zip_member_index,
    download_file,
    ensure_dir,
    extract_zip_file,
    load_manifest,
    save_manifest,
    stream_csv_sample,
)

HF_SHARD_PATTERN = re.compile(r"images/(?P<split>train|test)/(?P<file>[0-9]{2}\.zip)")


@dataclass
class PrepareResult:
    provider: str
    manifest_path: str
    image_root: str
    metadata_path: str | None = None
    archives: list[str] | None = None


def _normalize_manifest(
    frame: pd.DataFrame,
    *,
    split: str,
    coarse_bins: tuple[int, int],
    fine_bins: tuple[int, int],
    latitude_col: str,
    longitude_col: str,
    image_id_col: str,
    image_relpath_col: str,
) -> pd.DataFrame:
    frame = frame.copy()
    frame["latitude"] = frame[latitude_col].astype(float)
    frame["longitude"] = frame[longitude_col].astype(float)
    frame["image_id"] = frame[image_id_col].astype(str)
    frame["image_relpath"] = frame[image_relpath_col].astype(str)
    frame["split"] = split

    if "quadtree_10_5000" in frame.columns:
        frame["coarse_cell"] = frame["quadtree_10_5000"].astype(str)
    else:
        frame["coarse_cell"] = assign_grid_cells(
            frame["latitude"], frame["longitude"], coarse_bins
        )
    if "quadtree_10_1000" in frame.columns:
        frame["fine_cell"] = frame["quadtree_10_1000"].astype(str)
    else:
        frame["fine_cell"] = assign_grid_cells(frame["latitude"], frame["longitude"], fine_bins)

    keep = [
        "image_id",
        "image_relpath",
        "latitude",
        "longitude",
        "split",
        "coarse_cell",
        "fine_cell",
    ]
    for optional in (
        "country",
        "region",
        "sub-region",
        "city",
        "title",
        "grid_reference",
        "land_cover",
        "road_index",
        "drive_side",
        "climate",
        "soil",
        "dist_sea",
        "captured_at",
        "sequence",
        "unique_region",
        "unique_sub-region",
        "unique_city",
        "unique_country",
    ):
        if optional in frame.columns:
            keep.append(optional)
    return frame[keep]


class OSV5MProvider:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.provider_root = ensure_dir(config["data"]["provider_root"])
        self.raw_root = ensure_dir(self.provider_root / Path("raw"))
        self.manifest_root = ensure_dir(self.provider_root / Path("manifests"))
        self.metadata_base = config["download"]["osv5m_metadata_base"]
        self.tree_base = config["download"]["osv5m_tree_base"]

    def _manifest_path(self, split: str, max_rows: int | None) -> Path:
        return self.manifest_root / (
            f"{split}_head_{max_rows}.parquet" if max_rows is not None else f"{split}.parquet"
        )

    def _metadata_path(self, split: str, max_rows: int | None) -> Path:
        return self.raw_root / (
            f"{split}_head_{max_rows}.csv" if max_rows is not None else f"{split}.csv"
        )

    def _manifest_is_fresh(self, manifest_path: Path, metadata_path: Path) -> bool:
        if not manifest_path.exists() or not metadata_path.exists():
            return False
        manifest_mtime = manifest_path.stat().st_mtime
        dependency_mtime = metadata_path.stat().st_mtime
        index_path = self.raw_root / "zip_member_index.parquet"
        if index_path.exists():
            dependency_mtime = max(dependency_mtime, index_path.stat().st_mtime)
        return manifest_mtime >= dependency_mtime

    def _join_archive_locations(self, frame: pd.DataFrame) -> pd.DataFrame:
        index_path = self.raw_root / "zip_member_index.parquet"
        if not index_path.exists():
            return frame
        index_frame = pd.read_parquet(index_path)
        required_columns = {"filename", "archive_path", "member_name"}
        if not required_columns.issubset(index_frame.columns):
            return frame
        index_frame = index_frame[["filename", "archive_path", "member_name"]]
        enriched = frame.merge(
            index_frame,
            how="left",
            left_on="image_relpath",
            right_on="filename",
        )
        if "filename" in enriched.columns:
            enriched = enriched.drop(columns=["filename"])
        if "member_name" in enriched.columns:
            enriched = enriched.rename(columns={"member_name": "archive_member_name"})
        return enriched

    def list_shards(self, split: str) -> list[str]:
        url = f"{self.tree_base}/images/{split}"
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        matches = {
            f"images/{match.group('split')}/{match.group('file')}"
            for match in HF_SHARD_PATTERN.finditer(response.text)
            if match.group("split") == split
        }
        return sorted(matches)

    def download_metadata(self, split: str, max_rows: int | None = None) -> Path:
        metadata_url = f"{self.metadata_base}/{split}.csv?download=true"
        if max_rows is None:
            destination = self.raw_root / f"{split}.csv"
            return download_file(
                metadata_url,
                destination,
                chunk_size_mb=self.config["download"].get("chunk_size_mb", 4),
            )
        full_metadata_path = self.raw_root / f"{split}.csv"
        destination = self.raw_root / f"{split}_head_{max_rows}.csv"
        if destination.exists():
            return destination
        if full_metadata_path.exists():
            frame = pd.read_csv(full_metadata_path, nrows=max_rows)
            frame.to_csv(destination, index=False)
            return destination
        return stream_csv_sample(
            metadata_url,
            destination,
            max_rows=max_rows,
            chunk_size_mb=self.config["download"].get("chunk_size_mb", 1),
        )

    def download_shards(
        self,
        split: str,
        *,
        limit: int | None = None,
        shard_names: list[str] | None = None,
    ) -> list[Path]:
        archives_dir = ensure_dir(self.raw_root / "archives" / split)
        raw_extract_root = self.raw_root
        shard_names = shard_names or self.list_shards(split)
        if limit is not None:
            shard_names = shard_names[:limit]

        downloaded: list[Path] = []
        for shard_name in shard_names:
            shard_url = f"{self.metadata_base}/{shard_name}?download=true"
            destination = archives_dir / Path(shard_name).name
            archive_path = download_file(
                shard_url,
                destination,
                chunk_size_mb=self.config["download"].get("chunk_size_mb", 4),
            )
            downloaded.append(archive_path)
            if self.config["download"].get("extract_archives", True):
                extract_zip_file(archive_path, raw_extract_root)
                if self.config["download"].get("cleanup_archives", False):
                    archive_path.unlink(missing_ok=True)
        return downloaded

    def prepare(
        self,
        *,
        split: str = "train",
        max_rows: int | None = None,
        download_images: bool = False,
        shard_limit: int | None = None,
        force_rebuild: bool = False,
    ) -> PrepareResult:
        manifest_path = self._manifest_path(split, max_rows)
        metadata_path = self._metadata_path(split, max_rows)
        if not force_rebuild and self._manifest_is_fresh(manifest_path, metadata_path):
            return PrepareResult(
                provider="osv5m",
                manifest_path=str(manifest_path),
                image_root=str(self.raw_root),
                metadata_path=str(metadata_path),
                archives=[],
            )

        metadata_path = self.download_metadata(split, max_rows=max_rows)
        archives: list[str] = []
        if download_images:
            archives = [str(path) for path in self.download_shards(split, limit=shard_limit)]
        elif bool(self.config["download"].get("index_archives", False)):
            build_zip_member_index(
                self.raw_root,
                self.raw_root / "zip_member_index.parquet",
                max_workers=int(self.config["download"].get("index_max_workers", 8)),
            )

        frame = pd.read_csv(metadata_path)
        frame["image_relpath"] = frame["id"].astype("int64").astype(str) + ".jpg"
        manifest = _normalize_manifest(
            frame,
            split=split,
            coarse_bins=tuple(self.config["data"]["coarse_bins"]),
            fine_bins=tuple(self.config["data"]["fine_bins"]),
            latitude_col="latitude",
            longitude_col="longitude",
            image_id_col="id",
            image_relpath_col="image_relpath",
        )
        manifest = self._join_archive_locations(manifest)
        save_manifest(manifest, manifest_path)

        return PrepareResult(
            provider="osv5m",
            manifest_path=str(manifest_path),
            image_root=str(self.raw_root),
            metadata_path=str(metadata_path),
            archives=archives,
        )


class GeographSampleProvider:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.provider_root = ensure_dir(config["data"]["provider_root"])
        self.raw_root = ensure_dir(self.provider_root / Path("raw"))
        self.manifest_root = ensure_dir(self.provider_root / Path("manifests"))
        self.sample_url = config["download"]["geograph_sample_url"]

    def prepare(self, *, max_rows: int | None = None) -> PrepareResult:
        manifest_path = self.manifest_root / "geograph_sample.parquet"
        if manifest_path.exists():
            return PrepareResult(
                provider="geograph_sample",
                manifest_path=str(manifest_path),
                image_root=str(self.raw_root),
                metadata_path=str(self.raw_root / "geograph_dataset001.metadata.csv"),
                archives=[str(self.raw_root / "geograph_dataset001-sample.zip")],
            )
        archive_path = download_file(
            self.sample_url,
            self.raw_root / "geograph_dataset001-sample.zip",
            chunk_size_mb=self.config["download"].get("chunk_size_mb", 2),
        )
        extract_zip_file(archive_path, self.raw_root)
        metadata_path = self.raw_root / "geograph_dataset001.metadata.csv"
        frame = pd.read_csv(metadata_path)
        if max_rows is not None:
            frame = frame.head(max_rows).copy()
        manifest = _normalize_manifest(
            frame,
            split="train",
            coarse_bins=tuple(self.config["data"]["coarse_bins"]),
            fine_bins=tuple(self.config["data"]["fine_bins"]),
            latitude_col="wgs84_lat",
            longitude_col="wgs84_long",
            image_id_col="gridimage_id",
            image_relpath_col="filename",
        )
        save_manifest(manifest, manifest_path)
        return PrepareResult(
            provider="geograph_sample",
            manifest_path=str(manifest_path),
            image_root=str(self.raw_root),
            metadata_path=str(metadata_path),
            archives=[str(archive_path)],
        )


def get_provider(config: dict[str, Any]) -> OSV5MProvider | GeographSampleProvider:
    provider = config["data"]["provider"]
    if provider == "osv5m":
        return OSV5MProvider(config)
    if provider == "geograph_sample":
        return GeographSampleProvider(config)
    raise ValueError(f"Unsupported provider: {provider}")
