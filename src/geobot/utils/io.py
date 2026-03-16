from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import requests


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_file(
    url: str,
    destination: str | Path,
    *,
    chunk_size_mb: int = 4,
    overwrite: bool = False,
) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        return destination

    chunk_size = chunk_size_mb * 1024 * 1024
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    handle.write(chunk)
    return destination


def stream_csv_sample(
    url: str,
    destination: str | Path,
    *,
    max_rows: int,
    chunk_size_mb: int = 1,
    overwrite: bool = False,
) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        return destination

    lines: list[str] = []
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        for raw_line in response.iter_lines(decode_unicode=True):
            if raw_line is None:
                continue
            lines.append(raw_line)
            if len(lines) >= max_rows + 1:
                break
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def extract_zip_file(zip_path: str | Path, destination: str | Path) -> None:
    zip_path = Path(zip_path)
    destination = ensure_dir(destination)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(destination)


def load_manifest(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def save_manifest(frame: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)
    return path


def write_json(payload: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    return path


def sample_zip_member_lines(url: str, member_name: str, max_lines: int = 5) -> list[str]:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        with archive.open(member_name) as handle:
            text = io.TextIOWrapper(handle, encoding="utf-8")
            lines: list[str] = []
            for index, line in enumerate(text):
                lines.append(line.rstrip("\n"))
                if index + 1 >= max_lines:
                    break
    return lines
