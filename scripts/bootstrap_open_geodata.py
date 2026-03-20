from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[1]

USER_AGENT = "geo-autolab-bootstrap/0.1"

DEFAULT_ANCHORS: dict[str, tuple[float, float]] = {
    "chicago": (41.8781, -87.6298),
    "new_york": (40.7128, -74.0060),
    "san_francisco": (37.7749, -122.4194),
    "london": (51.5072, -0.1276),
}


def geocell_key(lat: float, lon: float, bin_size_deg: float = 2.0) -> tuple[int, int]:
    lat_bin = math.floor((lat + 90.0) / bin_size_deg)
    lon_bin = math.floor((lon + 180.0) / bin_size_deg)
    return lat_bin, lon_bin


def stable_split(group_id: str, val_ratio: float = 0.2) -> str:
    digest = hashlib.sha1(group_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "val" if bucket < val_ratio else "train"


def assign_splits(rows: list[dict[str, Any]], val_ratio: float = 0.2) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["source"], row["city"]), []).append(row)

    for (_source, _city), city_rows in grouped.items():
        by_group: dict[str, list[dict[str, Any]]] = {}
        for row in city_rows:
            by_group.setdefault(str(row["group_id"]), []).append(row)

        if len(by_group) > 1:
            ordered_groups = sorted(by_group.items(), key=lambda item: len(item[1]), reverse=True)
            val_groups = max(1, round(len(ordered_groups) * val_ratio))
            for group_index, (_group_id, group_rows) in enumerate(ordered_groups):
                split = "val" if group_index < val_groups else "train"
                for row in group_rows:
                    row["split"] = split
        else:
            only_group_rows = next(iter(by_group.values()))
            only_group_rows.sort(key=lambda row: str(row["image_id"]))
            val_count = max(1, round(len(only_group_rows) * val_ratio))
            split_index = len(only_group_rows) - val_count
            for index, row in enumerate(only_group_rows):
                row["split"] = "val" if index >= split_index else "train"
    return rows


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def download_binary(session: requests.Session, url: str, target: Path) -> bool:
    ensure_parent(target)
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            response = session.get(url, timeout=60, stream=True)
            if response.status_code in {429, 500, 502, 503, 504}:
                retry_after = response.headers.get("Retry-After")
                wait_seconds = float(retry_after) if retry_after else (1.5 * (attempt + 1))
                time.sleep(wait_seconds)
                continue
            response.raise_for_status()
            with target.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        handle.write(chunk)
            return True
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    if last_error is not None:
        raise last_error
    return True


def fetch_kartaview(
    session: requests.Session,
    lat: float,
    lon: float,
    page: int,
    ipp: int,
    radius: int,
) -> list[dict[str, Any]]:
    response = session.post(
        "https://api.openstreetcam.org/1.0/list/nearby-photos/",
        data={
            "lat": f"{lat:.6f}",
            "lng": f"{lon:.6f}",
            "radius": str(radius),
            "page": str(page),
            "ipp": str(ipp),
        },
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    return list(payload.get("currentPageItems", []))


def fetch_wikimedia(session: requests.Session, lat: float, lon: float, limit: int, radius: int) -> list[dict[str, Any]]:
    response = session.get(
        "https://commons.wikimedia.org/w/api.php",
        params={
            "action": "query",
            "generator": "geosearch",
            "ggsprimary": "all",
            "ggsnamespace": "6",
            "ggscoord": f"{lat:.6f}|{lon:.6f}",
            "ggsradius": str(radius),
            "ggslimit": str(limit),
            "prop": "coordinates|imageinfo",
            "iiprop": "url|extmetadata",
            "iiurlwidth": "1024",
            "format": "json",
        },
        timeout=60,
    )
    response.raise_for_status()
    pages = (response.json().get("query") or {}).get("pages") or {}
    return list(pages.values())


def build_rows_from_kartaview(
    session: requests.Session,
    output_dir: Path,
    anchors: dict[str, tuple[float, float]],
    per_anchor: int,
    radius: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    geocells: dict[tuple[int, int], int] = {}
    seen_ids: set[str] = set()

    for city_name, (lat, lon) in anchors.items():
        collected = 0
        page = 1
        while collected < per_anchor:
            items = fetch_kartaview(session, lat=lat, lon=lon, page=page, ipp=min(20, per_anchor * 2), radius=radius)
            if not items:
                break
            for item in items:
                image_id = str(item["id"])
                if image_id in seen_ids:
                    continue
                seen_ids.add(image_id)
                image_url = f"https://openstreetcam.org/{str(item['name']).lstrip('/')}"
                local_path = output_dir / "images" / "kartaview" / city_name / f"{image_id}.jpg"
                download_binary(session, image_url, local_path)

                image_lat = float(item["lat"])
                image_lon = float(item["lng"])
                cell_key = geocell_key(image_lat, image_lon)
                geocell_id = geocells.setdefault(cell_key, len(geocells))
                sequence_id = str(item.get("sequence_id", image_id))
                rows.append(
                    {
                        "image_path": str(local_path.relative_to(ROOT)).replace("\\", "/"),
                        "latitude": image_lat,
                        "longitude": image_lon,
                        "geocell_id": geocell_id,
                        "split": "",
                        "domain": "street",
                        "source": "kartaview",
                        "city": city_name,
                        "group_id": sequence_id,
                        "image_id": image_id,
                        "url": image_url,
                    }
                )
                collected += 1
                if collected >= per_anchor:
                    break
            page += 1
            time.sleep(0.25)
    return rows


def build_rows_from_wikimedia(
    session: requests.Session,
    output_dir: Path,
    anchors: dict[str, tuple[float, float]],
    per_anchor: int,
    radius: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    geocells: dict[tuple[int, int], int] = {}
    seen_ids: set[str] = set()

    for city_name, (lat, lon) in anchors.items():
        pages = fetch_wikimedia(session, lat=lat, lon=lon, limit=max(per_anchor * 2, 10), radius=radius)
        collected = 0
        for page in pages:
            page_id = str(page["pageid"])
            if page_id in seen_ids:
                continue
            seen_ids.add(page_id)
            image_info = (page.get("imageinfo") or [{}])[0]
            image_url = image_info.get("thumburl") or image_info.get("url")
            coords = (page.get("coordinates") or [{}])[0]
            if not image_url or "svg" in str(image_url).lower():
                continue
            image_lat = float(coords["lat"])
            image_lon = float(coords["lon"])
            local_path = output_dir / "images" / "wikimedia" / city_name / f"{page_id}.jpg"
            download_binary(session, image_url, local_path)

            cell_key = geocell_key(image_lat, image_lon)
            geocell_id = geocells.setdefault(cell_key, len(geocells))
            rows.append(
                {
                    "image_path": str(local_path.relative_to(ROOT)).replace("\\", "/"),
                    "latitude": image_lat,
                    "longitude": image_lon,
                    "geocell_id": geocell_id,
                    "split": "",
                    "domain": "open-photo",
                    "source": "wikimedia_commons",
                    "city": city_name,
                    "group_id": page_id,
                    "image_id": page_id,
                    "url": image_url,
                    "description_url": image_info.get("descriptionurl", ""),
                    "license": (((image_info.get("extmetadata") or {}).get("LicenseShortName") or {}).get("value")) or "",
                }
            )
            collected += 1
            if collected >= per_anchor:
                break
            time.sleep(0.25)
    return rows


def write_manifest(output_dir: Path, rows: list[dict[str, Any]]) -> Path:
    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", "latitude", "longitude", "geocell_id", "split", "domain", "source"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})
    return manifest_path


def write_metadata(output_dir: Path, rows: list[dict[str, Any]]) -> Path:
    metadata_path = output_dir / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return metadata_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a tiny open geolocation dataset for Geo AutoLab.")
    parser.add_argument("--source", choices=["kartaview", "wikimedia", "mixed"], default="kartaview")
    parser.add_argument("--per-anchor", type=int, default=6, help="Images to download per anchor city and source")
    parser.add_argument("--radius", type=int, default=500, help="Search radius in meters")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "data"),
        help="Output directory that will receive manifest.csv and images/",
    )
    parser.add_argument(
        "--anchors",
        nargs="*",
        default=list(DEFAULT_ANCHORS.keys()),
        help="Anchor city names from the built-in set",
    )
    args = parser.parse_args()

    anchors = {name: DEFAULT_ANCHORS[name] for name in args.anchors if name in DEFAULT_ANCHORS}
    if not anchors:
        raise SystemExit("No valid anchors selected.")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    rows: list[dict[str, Any]] = []
    if args.source in {"kartaview", "mixed"}:
        rows.extend(
            build_rows_from_kartaview(
                session=session,
                output_dir=output_dir,
                anchors=anchors,
                per_anchor=args.per_anchor,
                radius=args.radius,
            )
        )
    if args.source in {"wikimedia", "mixed"}:
        rows.extend(
            build_rows_from_wikimedia(
                session=session,
                output_dir=output_dir,
                anchors=anchors,
                per_anchor=args.per_anchor,
                radius=max(args.radius, 800),
            )
        )

    rows = assign_splits(rows)
    random.Random(7).shuffle(rows)
    manifest_path = write_manifest(output_dir, rows)
    metadata_path = write_metadata(output_dir, rows)
    summary = {
        "source": args.source,
        "anchors": list(anchors.keys()),
        "rows": len(rows),
        "train_rows": sum(1 for row in rows if row["split"] == "train"),
        "val_rows": sum(1 for row in rows if row["split"] == "val"),
        "manifest_path": str(manifest_path),
        "metadata_path": str(metadata_path),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
