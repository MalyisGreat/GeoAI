from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def rel_to_root(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def capture_command(command: list[str], log_path: Path, cwd: Path) -> int:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
        return process.wait()


def maybe_capture(cmd: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(ROOT),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return completed.stdout.strip() or None


def build_resolved_config(
    base_config_path: Path,
    output_dir: Path,
    run_root: Path,
    *,
    max_train_samples: int | None,
    max_val_samples: int | None,
    max_epochs: int | None,
    batch_size: int | None,
    eval_batch_size: int | None,
    num_workers: int | None,
    eval_every_images: int | None,
) -> dict[str, Any]:
    with base_config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    train = config.setdefault("train", {})
    config["run_root"] = rel_to_root(run_root)
    config["history_path"] = rel_to_root(run_root / "history.jsonl")
    train["run_dir"] = rel_to_root(run_root / "manual")
    train["manifest_path"] = rel_to_root(output_dir / "manifest.csv")
    if max_train_samples is not None:
        train["max_train_samples"] = max_train_samples
    if max_val_samples is not None:
        train["max_val_samples"] = max_val_samples
    if max_epochs is not None:
        train["max_epochs"] = max_epochs
    if batch_size is not None:
        train["batch_size"] = batch_size
    if eval_batch_size is not None:
        train["eval_batch_size"] = eval_batch_size
    if num_workers is not None:
        train["num_workers"] = num_workers
    if eval_every_images is not None:
        train["eval_every_images"] = eval_every_images
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description="Download an OSV5M subset and train the A100 config with logs.")
    parser.add_argument(
        "--config",
        default="configs/autonomy/osv5m_200k_convnext_small_hierarchical_224_a100.yaml",
        help="Base AutoLab YAML config",
    )
    parser.add_argument("--output-dir", default="data/osv5m_200k_a100", help="Output dataset directory")
    parser.add_argument("--shards", nargs="+", default=["00", "01", "02", "03"], help="OSV5M train shards")
    parser.add_argument("--limit-per-shard", type=int, default=50000, help="Images per shard")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Override train sample cap")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Override val sample cap")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--batch-size", type=int, default=None, help="Override train batch size")
    parser.add_argument("--eval-batch-size", type=int, default=None, help="Override eval batch size")
    parser.add_argument("--num-workers", type=int, default=None, help="Override dataloader workers")
    parser.add_argument("--eval-every-images", type=int, default=None, help="Override periodic eval interval")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter to use")
    parser.add_argument("--skip-download", action="store_true", help="Skip subset download and reuse existing manifest")
    parser.add_argument("--skip-train", action="store_true", help="Skip training after config resolution")
    parser.add_argument("--dry-run", action="store_true", help="Write logs and config, but do not run child commands")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = ensure_dir(ROOT / "logs" / "a100" / timestamp)
    run_root = ensure_dir(ROOT / "runs" / "a100" / timestamp)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    base_config_path = Path(args.config)
    if not base_config_path.is_absolute():
        base_config_path = ROOT / base_config_path

    resolved_config = build_resolved_config(
        base_config_path=base_config_path,
        output_dir=output_dir,
        run_root=run_root,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        eval_every_images=args.eval_every_images,
    )
    resolved_config_path = log_dir / "resolved-config.yaml"
    with resolved_config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved_config, handle, sort_keys=False)

    download_cmd = [
        args.python,
        str(ROOT / "scripts" / "prepare_osv5m_subset.py"),
        "--output-dir",
        str(output_dir),
        "--limit-per-shard",
        str(args.limit_per_shard),
        "--log-every",
        "5000",
        "--shards",
        *args.shards,
    ]
    train_cmd = [
        args.python,
        str(ROOT / "scripts" / "train_once.py"),
        str(resolved_config_path),
    ]

    startup = {
        "timestamp": timestamp,
        "cwd": str(ROOT),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": args.python,
        "config": str(base_config_path),
        "resolved_config": str(resolved_config_path),
        "output_dir": str(output_dir),
        "run_root": str(run_root),
        "shards": args.shards,
        "limit_per_shard": args.limit_per_shard,
        "git_commit": maybe_capture(["git", "rev-parse", "HEAD"]),
        "nvidia_smi": maybe_capture(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
    }
    (log_dir / "startup.json").write_text(json.dumps(startup, indent=2), encoding="utf-8")
    (log_dir / "commands.json").write_text(
        json.dumps(
            {
                "download": download_cmd,
                "train": train_cmd,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "event": "a100_launcher_start",
                "timestamp": timestamp,
                "log_dir": str(log_dir),
                "run_root": str(run_root),
                "output_dir": str(output_dir),
                "dry_run": args.dry_run,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    if args.dry_run:
        print(json.dumps({"event": "a100_launcher_dry_run", "download": download_cmd, "train": train_cmd}, sort_keys=True), flush=True)
        return 0

    if not args.skip_download:
        exit_code = capture_command(download_cmd, log_dir / "download-console.log", ROOT)
        if exit_code != 0:
            return exit_code

    if not args.skip_train:
        exit_code = capture_command(train_cmd, log_dir / "train-console.log", ROOT)
        if exit_code != 0:
            return exit_code

    print(json.dumps({"event": "a100_launcher_complete", "log_dir": str(log_dir), "run_root": str(run_root)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
