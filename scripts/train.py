from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from _bootstrap import bootstrap_project_src

bootstrap_project_src(PROJECT_ROOT)

from geobot.config import load_config
from geobot.data import get_provider
from geobot.train import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a hybrid geolocation model.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--download-images", action="store_true")
    parser.add_argument("--shard-limit", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    train_manifest = Path(config["data"]["train_manifest"])
    prepare_max_rows = args.max_rows
    if prepare_max_rows is None:
        prepare_max_rows = config["data"].get("max_train_rows")
    if args.prepare or not train_manifest.exists():
        provider = get_provider(config)
        if config["data"]["provider"] == "osv5m":
            result = provider.prepare(
                split="train",
                max_rows=prepare_max_rows,
                download_images=args.download_images,
                shard_limit=args.shard_limit,
            )
        else:
            result = provider.prepare(max_rows=prepare_max_rows)
        config["data"]["train_manifest"] = result.manifest_path
        config["data"]["image_root"] = result.image_root

    summary = run_training(config)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
