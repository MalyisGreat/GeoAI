from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from geobot.config import load_config
from geobot.train.engine import _make_loaders, _prepare_frames


def run_variant(config: dict, *, steps: int) -> dict[str, object]:
    train_frame, val_frame, _ = _prepare_frames(config)
    train_ds, _, train_loader, _ = _make_loaders(train_frame, val_frame, config)
    total_images = 0
    started = time.perf_counter()
    for index, batch in enumerate(train_loader, start=1):
        total_images += int(batch["image"].shape[0])
        if index >= steps:
            break
    elapsed = time.perf_counter() - started
    return {
        "dataset": type(train_ds).__name__,
        "images": total_images,
        "seconds": round(elapsed, 3),
        "images_per_sec": round(total_images / max(elapsed, 1e-6), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark train dataloader variants.")
    parser.add_argument("--config", default="configs/local_rtx3060_20k.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--max-train-rows", type=int, default=4096)
    parser.add_argument("--max-val-rows", type=int, default=256)
    args = parser.parse_args()

    base = load_config(args.config)
    base["data"]["max_train_rows"] = args.max_train_rows
    base["data"]["max_val_rows"] = args.max_val_rows

    variants = [
        ("archive-w1", {"num_workers": 1, "worker_sharding_mode": "archive"}),
        ("archive-w8", {"num_workers": 8, "worker_sharding_mode": "archive"}),
        ("record-w8", {"num_workers": 8, "worker_sharding_mode": "record"}),
    ]
    results: list[dict[str, object]] = []
    for name, overrides in variants:
        config = deepcopy(base)
        config["train"].update(overrides)
        result = run_variant(config, steps=args.steps)
        result["variant"] = name
        result["workers"] = config["train"]["num_workers"]
        result["worker_sharding_mode"] = config["train"]["worker_sharding_mode"]
        results.append(result)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
