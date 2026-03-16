from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from geobot.config import load_config
from geobot.data import get_provider
from geobot.train import run_training


def main() -> None:
    config = load_config("configs/smoke.yaml")
    provider = get_provider(config)
    result = provider.prepare(max_rows=config["data"].get("max_train_rows"))
    config["data"]["train_manifest"] = result.manifest_path
    config["data"]["image_root"] = result.image_root
    summary = run_training(config)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
