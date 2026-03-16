from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from geobot.config import load_config
from geobot.data import get_provider


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and normalize public geolocation data.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--download-images", action="store_true")
    parser.add_argument("--shard-limit", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    provider = get_provider(config)

    if config["data"]["provider"] == "osv5m":
        result = provider.prepare(
            split=args.split,
            max_rows=args.max_rows,
            download_images=args.download_images,
            shard_limit=args.shard_limit,
        )
    else:
        result = provider.prepare(max_rows=args.max_rows)
    print(json.dumps(result.__dict__, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
