from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATH_KEYS = {
    "provider_root",
    "train_manifest",
    "val_manifest",
    "image_root",
    "checkpoint_dir",
    "metrics_dir",
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _absolutize_paths(node: Any) -> Any:
    if isinstance(node, dict):
        output: dict[str, Any] = {}
        for key, value in node.items():
            if key in PATH_KEYS and isinstance(value, str) and value:
                output[key] = str((PROJECT_ROOT / value).resolve())
            else:
                output[key] = _absolutize_paths(value)
        return output
    if isinstance(node, list):
        return [_absolutize_paths(item) for item in node]
    return node


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    config = _absolutize_paths(loaded)
    config["project_root"] = str(PROJECT_ROOT)
    return config


def save_config(config: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
