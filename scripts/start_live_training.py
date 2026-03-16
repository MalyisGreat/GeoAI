from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from geobot.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch live dashboard and a visible training window.")
    parser.add_argument("--config", default="configs/local_rtx3060_20k_live.yaml")
    parser.add_argument("--python-exe", default=str((PROJECT_ROOT / ".venv-cu312" / "Scripts" / "python.exe").resolve()))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--reset-run-dir", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    run_root = Path(config["project_root"]) / "runs" / config["experiment_name"]
    if args.reset_run_dir and run_root.exists():
        shutil.rmtree(run_root)

    logs_dir = PROJECT_ROOT / "logs" / config["experiment_name"]
    logs_dir.mkdir(parents=True, exist_ok=True)
    dashboard_log = logs_dir / "dashboard.log"
    train_console_log = logs_dir / "train-console.log"

    dashboard_process = subprocess.Popen(
        [
            args.python_exe,
            str(PROJECT_ROOT / "scripts" / "live_training_dashboard.py"),
            "--run-root",
            str(run_root),
            "--port",
            str(args.port),
            "--open-browser",
        ],
        cwd=str(PROJECT_ROOT),
        stdout=dashboard_log.open("w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
    )
    time.sleep(1.5)

    train_command = (
        f"& '{args.python_exe}' '{PROJECT_ROOT / 'scripts' / 'train.py'}' --config '{Path(args.config).resolve()}' "
        f"2>&1 | Tee-Object -FilePath '{train_console_log}'"
    )
    creationflags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
    training_process = subprocess.Popen(
        [
            "powershell",
            "-NoExit",
            "-Command",
            train_command,
        ],
        cwd=str(PROJECT_ROOT),
        creationflags=creationflags,
    )

    print(f"dashboard_url=http://127.0.0.1:{args.port}/")
    print(f"dashboard_pid={dashboard_process.pid}")
    print(f"training_pid={training_process.pid}")
    print(f"run_root={run_root}")
    print(f"dashboard_log={dashboard_log}")
    print(f"train_console_log={train_console_log}")


if __name__ == "__main__":
    main()
