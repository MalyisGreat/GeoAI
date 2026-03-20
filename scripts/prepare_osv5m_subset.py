from __future__ import annotations

import argparse
import csv
import json
import time
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[1]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def log(event: str, **payload: object) -> None:
    row = {"event": event, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    row.update(payload)
    print(json.dumps(row, sort_keys=True), flush=True)


def extract_subset(zip_path: Path, image_dir: Path, limit: int, log_every: int) -> set[str]:
    ensure_dir(image_dir)
    extracted_ids: set[str] = set()
    extracted_count = 0
    skipped_existing = 0
    log("extract_start", zip_path=str(zip_path), image_dir=str(image_dir), limit=limit)
    with zipfile.ZipFile(zip_path) as archive:
        members = [m for m in archive.infolist() if not m.is_dir() and m.filename.lower().endswith(".jpg")]
        members.sort(key=lambda m: m.filename)
        for member in members[:limit]:
            target = image_dir / Path(member.filename).name
            if not target.exists():
                with archive.open(member) as src, target.open("wb") as dst:
                    dst.write(src.read())
                extracted_count += 1
                if log_every > 0 and extracted_count % log_every == 0:
                    log(
                        "extract_progress",
                        zip_path=str(zip_path),
                        extracted=extracted_count,
                        skipped_existing=skipped_existing,
                    )
            else:
                skipped_existing += 1
            extracted_ids.add(target.stem)
    log(
        "extract_complete",
        zip_path=str(zip_path),
        extracted=extracted_count,
        skipped_existing=skipped_existing,
        kept=len(extracted_ids),
    )
    return extracted_ids


def stable_split(sequence: str) -> str:
    value = int.from_bytes(sequence.encode("utf-8"), "little", signed=False) % 10
    return "val" if value == 0 else "train"


def load_existing_rows(path: Path | None) -> dict[str, dict[str, object]]:
    if path is None or not path.exists():
        return {}
    rows: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row["image_id"])] = row
    return rows


def build_existing_cell_map(csv_path: Path, existing_rows: dict[str, dict[str, object]]) -> dict[str, int]:
    if not existing_rows:
        return {}
    cell_map: dict[str, int] = {}
    wanted = set(existing_rows.keys())
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_id = row["id"]
            if image_id not in wanted:
                continue
            existing = existing_rows[image_id]
            cell_key = row["quadtree_10_2500"] or row["cell"] or f"{row['latitude']},{row['longitude']}"
            cell_map[cell_key] = int(existing["geocell_id"])
    return cell_map


def build_manifest(
    csv_path: Path,
    output_dir: Path,
    valid_ids: set[str],
    limit: int,
    existing_rows: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    rows: list[dict[str, str | int | float]] = []
    existing_rows = existing_rows or {}
    geocell_map = build_existing_cell_map(csv_path, existing_rows)
    next_geocell_id = (max(geocell_map.values()) + 1) if geocell_map else 0
    image_root = output_dir / "images"
    log(
        "manifest_build_start",
        csv_path=str(csv_path),
        output_dir=str(output_dir),
        valid_image_count=len(valid_ids),
        row_limit=limit,
    )

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_id = row["id"]
            if image_id not in valid_ids:
                continue
            cell_key = row["quadtree_10_2500"] or row["cell"] or f"{row['latitude']},{row['longitude']}"
            existing = existing_rows.get(image_id)
            if existing is not None:
                geocell_id = int(existing["geocell_id"])
                split = str(existing["split"])
            else:
                geocell_id = geocell_map.get(cell_key, next_geocell_id)
                if cell_key not in geocell_map:
                    geocell_map[cell_key] = geocell_id
                    next_geocell_id += 1
                split = stable_split(row["sequence"] or image_id)
            rows.append(
                {
                    "image_path": str((image_root / f"{image_id}.jpg").relative_to(ROOT)).replace("\\", "/"),
                    "latitude": float(row["latitude"]),
                    "longitude": float(row["longitude"]),
                    "geocell_id": geocell_id,
                    "split": split,
                    "domain": "street",
                    "source": "osv5m",
                    "country": row["country"],
                    "region": row["region"],
                    "sub_region": row["sub-region"],
                    "city": row["city"],
                    "sequence": row["sequence"],
                    "captured_at": row["captured_at"],
                    "thumb_original_url": row["thumb_original_url"],
                    "cell_key": cell_key,
                    "image_id": image_id,
                }
            )
            if len(rows) >= limit:
                break

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", "latitude", "longitude", "geocell_id", "split", "domain", "source"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})

    metadata_path = output_dir / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    summary = {
        "rows": len(rows),
        "train_rows": sum(1 for row in rows if row["split"] == "train"),
        "val_rows": sum(1 for row in rows if row["split"] == "val"),
        "geocell_classes": len(geocell_map),
        "manifest_path": str(manifest_path),
        "metadata_path": str(metadata_path),
        "image_dir": str(image_root),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log("manifest_build_complete", **summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare an OSV5M subset from one or more train shards.")
    parser.add_argument("--shards", nargs="+", default=["00"], help="Two-digit train shard ids")
    parser.add_argument("--limit-per-shard", type=int, default=50000, help="Number of images to keep per shard")
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "osv5m_50k"))
    parser.add_argument("--log-every", type=int, default=5000, help="Extraction progress interval")
    parser.add_argument(
        "--base-metadata",
        default=None,
        help="Optional existing metadata.jsonl used to preserve prior geocell ids and splits",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    ensure_dir(output_dir)
    log("subset_prepare_start", output_dir=str(output_dir), shards=args.shards, limit_per_shard=args.limit_per_shard)

    log("metadata_download_start", repo_id="osv5m/osv5m", filename="train.csv")
    csv_path = Path(hf_hub_download(repo_id="osv5m/osv5m", filename="train.csv", repo_type="dataset"))
    log("metadata_download_complete", csv_path=str(csv_path))
    image_dir = output_dir / "images"
    extracted_ids: set[str] = set()
    zip_paths: list[str] = []
    for shard in args.shards:
        log("shard_download_start", shard=shard, repo_id="osv5m/osv5m")
        zip_path = Path(
            hf_hub_download(
                repo_id="osv5m/osv5m",
                filename=f"{shard}.zip",
                subfolder="images/train",
                repo_type="dataset",
            )
        )
        log(
            "shard_download_complete",
            shard=shard,
            zip_path=str(zip_path),
            bytes=zip_path.stat().st_size,
        )
        zip_paths.append(str(zip_path))
        extracted_ids.update(
            extract_subset(
                zip_path=zip_path,
                image_dir=image_dir,
                limit=args.limit_per_shard,
                log_every=args.log_every,
            )
        )

    existing_rows = load_existing_rows(Path(args.base_metadata) if args.base_metadata else None)
    summary = build_manifest(
        csv_path=csv_path,
        output_dir=output_dir,
        valid_ids=extracted_ids,
        limit=args.limit_per_shard * len(args.shards),
        existing_rows=existing_rows,
    )
    summary["zip_paths"] = zip_paths
    summary["csv_path"] = str(csv_path)
    log("subset_prepare_complete", **summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
