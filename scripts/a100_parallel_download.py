from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from huggingface_hub import snapshot_download

from geobot.config import load_config
from geobot.data.providers import OSV5MProvider
from geobot.utils.io import append_jsonl, build_zip_member_index, ensure_dir


def log_event(log_path: Path, payload: dict) -> None:
    append_jsonl(log_path, {"ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **payload})


def configure_fast_hf() -> dict[str, str]:
    env_updates = {"HF_XET_HIGH_PERFORMANCE": "1"}
    if importlib.util.find_spec("hf_transfer") is not None:
        env_updates["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    for key, value in env_updates.items():
        os.environ.setdefault(key, value)
    return env_updates


def build_allow_patterns(provider: OSV5MProvider, splits: list[str], limit_shards: int | None, metadata_only: bool) -> list[str]:
    patterns = ["README.md"]
    for split in splits:
        patterns.append(f"{split}.csv")
        if metadata_only:
            continue
        if limit_shards is None:
            patterns.append(f"images/{split}/*.zip")
        else:
            for shard_name in provider.list_shards(split)[:limit_shards]:
                patterns.append(shard_name)
    return patterns


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast OSV-5M sync using Hugging Face snapshot download.")
    parser.add_argument("--config", default="configs/h100_atlas.yaml")
    parser.add_argument("--splits", nargs="+", default=["train"])
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--limit-shards", type=int, default=None)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-index", action="store_true")
    parser.add_argument("--force-manifest", action="store_true")
    parser.add_argument("--log-dir", default="logs/h100-sync")
    args = parser.parse_args()

    config = load_config(args.config)
    provider = OSV5MProvider(config)
    log_dir = ensure_dir(args.log_dir)
    log_path = log_dir / "sync.jsonl"

    env_updates = configure_fast_hf()
    max_workers = args.max_workers or int(config["download"].get("snapshot_max_workers", 16))
    allow_patterns = build_allow_patterns(provider, args.splits, args.limit_shards, args.metadata_only)

    plan = {
        "event": "sync-plan",
        "repo_id": "osv5m/osv5m",
        "splits": args.splits,
        "allow_patterns": allow_patterns,
        "local_dir": str(provider.raw_root),
        "max_workers": max_workers,
        "metadata_only": args.metadata_only,
        "dry_run": args.dry_run,
        "env": env_updates,
    }
    print(json.dumps(plan, indent=2, sort_keys=True))
    log_event(log_path, plan)

    started = time.perf_counter()
    dry_run_result = snapshot_download(
        repo_id="osv5m/osv5m",
        repo_type="dataset",
        local_dir=provider.raw_root,
        allow_patterns=allow_patterns,
        max_workers=max_workers,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        final = {
            "event": "dry-run-finished",
            "planned_files": len(dry_run_result),
            "log_path": str(log_path),
        }
        log_event(log_path, final)
        print(json.dumps(final, indent=2, sort_keys=True))
        return
    log_event(
        log_path,
        {
            "event": "snapshot-complete",
            "seconds": round(time.perf_counter() - started, 3),
            "local_dir": str(provider.raw_root),
        },
    )

    index_path = None
    if not args.metadata_only and bool(config["download"].get("index_archives", True)):
        index_started = time.perf_counter()
        index_path = build_zip_member_index(
            provider.raw_root,
            provider.raw_root / "zip_member_index.parquet",
            max_workers=int(config["download"].get("index_max_workers", max_workers)),
            force=args.force_index,
        )
        log_event(
            log_path,
            {
                "event": "index-complete",
                "seconds": round(time.perf_counter() - index_started, 3),
                "index_path": str(index_path),
            },
        )

    built_manifests = []
    manifest_started = time.perf_counter()
    for split in args.splits:
        result = provider.prepare(
            split=split,
            max_rows=None,
            download_images=False,
            force_rebuild=args.force_manifest,
        )
        built_manifests.append(result.__dict__)
    log_event(
        log_path,
        {
            "event": "manifest-complete",
            "seconds": round(time.perf_counter() - manifest_started, 3),
            "manifests": built_manifests,
        },
    )

    final = {
        "event": "sync-finished",
        "manifest_results": built_manifests,
        "zip_member_index": str(index_path) if index_path is not None else None,
        "log_path": str(log_path),
    }
    log_event(log_path, final)
    print(json.dumps(final, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
