from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from geobot.config import load_config
from geobot.data.providers import OSV5MProvider
from geobot.utils.io import append_jsonl, download_file, ensure_dir, extract_zip_file


def log_event(log_path: Path, payload: dict) -> None:
    append_jsonl(log_path, {"ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **payload})


def download_metadata(provider: OSV5MProvider, split: str, log_path: Path) -> Path:
    started = time.perf_counter()
    log_event(log_path, {"phase": "metadata", "split": split, "status": "start"})
    path = provider.download_metadata(split, max_rows=None)
    log_event(
        log_path,
        {
            "phase": "metadata",
            "split": split,
            "status": "done",
            "path": str(path),
            "seconds": round(time.perf_counter() - started, 3),
        },
    )
    return path


def download_and_extract_shard(
    provider: OSV5MProvider,
    shard_name: str,
    *,
    log_path: Path,
) -> dict:
    started = time.perf_counter()
    split = shard_name.split("/")[1]
    archives_dir = ensure_dir(provider.raw_root / "archives" / split)
    archive_path = archives_dir / Path(shard_name).name
    marker_dir = ensure_dir(provider.raw_root / ".extract_markers" / split)
    marker_path = marker_dir / f"{archive_path.stem}.done"
    previously_downloaded = archive_path.exists()
    previously_extracted = marker_path.exists()

    log_event(
        log_path,
        {
            "phase": "shard",
            "split": split,
            "shard": shard_name,
            "status": "start",
            "already_downloaded": previously_downloaded,
            "already_extracted": previously_extracted,
        },
    )

    download_file(
        f"{provider.metadata_base}/{shard_name}?download=true",
        archive_path,
        chunk_size_mb=provider.config["download"].get("chunk_size_mb", 16),
    )
    if provider.config["download"].get("extract_archives", True) and not previously_extracted:
        extract_zip_file(archive_path, provider.raw_root)
        marker_path.write_text("ok\n", encoding="utf-8")

    elapsed = time.perf_counter() - started
    payload = {
        "phase": "shard",
        "split": split,
        "shard": shard_name,
        "status": "done",
        "archive_path": str(archive_path),
        "size_bytes": archive_path.stat().st_size,
        "seconds": round(elapsed, 3),
    }
    log_event(log_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel OSV-5M downloader for A100 training.")
    parser.add_argument("--config", default="configs/a100_full.yaml")
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--limit-shards", type=int, default=None)
    parser.add_argument("--log-dir", default="logs/a100-download")
    args = parser.parse_args()

    config = load_config(args.config)
    provider = OSV5MProvider(config)
    log_dir = ensure_dir(args.log_dir)
    log_path = log_dir / "download.jsonl"

    metadata_paths = {}
    for split in args.splits:
        metadata_paths[split] = str(download_metadata(provider, split, log_path))

    shard_names: list[str] = []
    for split in args.splits:
        split_shards = provider.list_shards(split)
        if args.limit_shards is not None:
            split_shards = split_shards[: args.limit_shards]
        shard_names.extend(split_shards)

    summary = {
        "max_workers": args.max_workers,
        "metadata_paths": metadata_paths,
        "num_shards": len(shard_names),
        "provider_root": str(provider.provider_root),
    }
    print(json.dumps({"event": "download-plan", **summary}, indent=2, sort_keys=True))
    log_event(log_path, {"phase": "plan", **summary})

    completed = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(download_and_extract_shard, provider, shard_name, log_path=log_path): shard_name
            for shard_name in shard_names
        }
        for future in as_completed(futures):
            result = future.result()
            completed.append(result)
            print(json.dumps(result, sort_keys=True))

    final = {
        "event": "download-finished",
        "completed_shards": len(completed),
        "log_path": str(log_path),
    }
    log_event(log_path, final)
    print(json.dumps(final, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
