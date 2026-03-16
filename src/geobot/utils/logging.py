from __future__ import annotations

import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import append_jsonl, ensure_dir, write_json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_run_dirs(project_root: str | Path, experiment_name: str) -> dict[str, Path]:
    root = ensure_dir(Path(project_root) / "runs" / experiment_name)
    metrics = ensure_dir(root / "metrics")
    checkpoints = ensure_dir(root / "checkpoints")
    artifacts = ensure_dir(root / "artifacts")
    return {
        "root": root,
        "metrics": metrics,
        "checkpoints": checkpoints,
        "artifacts": artifacts,
    }


def log_event(path: str | Path, payload: dict[str, Any]) -> None:
    append_jsonl(path, {"timestamp_utc": utc_now(), **payload})


def save_summary(path: str | Path, payload: dict[str, Any]) -> None:
    write_json({"timestamp_utc": utc_now(), **payload}, path)


def system_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "python": sys.executable,
        "nvidia_smi": None,
    }
    if shutil.which("nvidia-smi"):
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            snapshot["nvidia_smi"] = completed.stdout.strip().splitlines()
        except Exception:
            snapshot["nvidia_smi"] = "unavailable"
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        snapshot["ram_total_gb"] = round(vm.total / (1024**3), 2)
        snapshot["ram_available_gb"] = round(vm.available / (1024**3), 2)
        snapshot["cpu_percent"] = psutil.cpu_percent(interval=0.1)
    except Exception:
        snapshot["ram_total_gb"] = None
        snapshot["ram_available_gb"] = None
        snapshot["cpu_percent"] = None
    return snapshot
