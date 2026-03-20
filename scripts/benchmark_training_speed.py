from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from geo_autolab.config import load_yaml
from geo_autolab.config_utils import append_jsonl
from geo_autolab.models import ModelConfig, build_model_stack
from geo_autolab.train.config import TrainConfig
from geo_autolab.train.dataset import build_dataloaders
from geo_autolab.train.engine import build_optimizer, build_scheduler
from geo_autolab.utils import choose_device, seed_everything, to_device


def apply_runtime_settings(train_config: TrainConfig, model: torch.nn.Module, device: torch.device) -> None:
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = train_config.cudnn_benchmark
    model.to(device)
    if train_config.channels_last and device.type == "cuda":
        model.to(memory_format=torch.channels_last)


def benchmark_once(bench_config: dict[str, Any]) -> dict[str, Any]:
    model_config_path = bench_config["model_config_path"]
    model_config = ModelConfig.model_validate(load_yaml(model_config_path))
    train_config = TrainConfig.model_validate(bench_config["train"])
    time_budget_sec = float(bench_config.get("time_budget_sec", 30))
    warmup_steps = int(bench_config.get("warmup_steps", 5))
    max_steps = int(bench_config.get("max_steps", 200))

    manifest_path = Path(train_config.manifest_path)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    train_config.manifest_path = str(manifest_path)

    if model_config.head.geocell_classes <= 0:
        max_geocell = -1
        import csv

        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                max_geocell = max(max_geocell, int(row["geocell_id"]))
        model_config.head.geocell_classes = max_geocell + 1

    seed_everything(train_config.seed)
    device = choose_device()
    model, criterion = build_model_stack(model_config)
    apply_runtime_settings(train_config, model, device)
    criterion.to(device)

    train_loader, _val_loader = build_dataloaders(model_config.image_size, train_config)
    optimizer = build_optimizer(model, train_config)
    scheduler = build_scheduler(optimizer, train_config, steps_per_epoch=max(1, len(train_loader)))
    scaler = torch.amp.GradScaler("cuda", enabled=train_config.amp and device.type == "cuda") if device.type == "cuda" else None

    iterator = iter(train_loader)
    model.train()
    optimizer.zero_grad(set_to_none=True)

    losses: list[float] = []
    measured_images = 0
    measured_steps = 0
    total_steps = 0
    start_time = None

    while total_steps < max_steps:
        if start_time is not None and (time.perf_counter() - start_time) >= time_budget_sec:
            break
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)

        batch = to_device(batch, device)
        if train_config.channels_last and isinstance(batch.get("image"), torch.Tensor):
            batch["image"] = batch["image"].contiguous(memory_format=torch.channels_last)

        with torch.autocast(device_type=device.type, enabled=train_config.amp and device.type == "cuda"):
            outputs = model(batch["image"])
            loss_breakdown = criterion(outputs, batch)
            loss = loss_breakdown.total / train_config.grad_accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        should_step = (total_steps + 1) % train_config.grad_accum_steps == 0
        if should_step:
            if scaler is not None:
                scale_before = scaler.get_scale()
                scaler.unscale_(optimizer)
            else:
                scale_before = None
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()
            else:
                optimizer.step()
                scale_after = None
            optimizer.zero_grad(set_to_none=True)
            if scaler is None or (scale_before is not None and scale_after >= scale_before):
                scheduler.step()

        total_steps += 1
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        loss_value = float(loss_breakdown.total.item())
        losses.append(loss_value)

        if total_steps == warmup_steps:
            start_time = time.perf_counter()
            measured_images = 0
            measured_steps = 0
            continue
        if total_steps > warmup_steps:
            batch_size = int(batch["image"].shape[0])
            measured_images += batch_size
            measured_steps += 1

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory_mb = float(torch.cuda.max_memory_allocated(device) / (1024**2))
    else:
        peak_memory_mb = 0.0

    if start_time is None:
        elapsed_sec = 0.0
    else:
        elapsed_sec = max(1e-6, time.perf_counter() - start_time)

    measured_losses = losses[warmup_steps:] if len(losses) > warmup_steps else losses
    start_loss = measured_losses[0] if measured_losses else None
    end_loss = measured_losses[-1] if measured_losses else None
    mean_loss = sum(measured_losses) / len(measured_losses) if measured_losses else None

    return {
        "images_per_sec": measured_images / elapsed_sec if elapsed_sec > 0 else 0.0,
        "optimizer_steps_per_sec": measured_steps / elapsed_sec if elapsed_sec > 0 else 0.0,
        "elapsed_sec": elapsed_sec,
        "measured_images": measured_images,
        "measured_steps": measured_steps,
        "total_steps": total_steps,
        "start_loss": start_loss,
        "end_loss": end_loss,
        "loss_delta": (end_loss - start_loss) if start_loss is not None and end_loss is not None else None,
        "loss_ratio": (end_loss / start_loss) if start_loss not in (None, 0.0) and end_loss is not None else None,
        "mean_loss": mean_loss,
        "peak_memory_mb": peak_memory_mb,
        "batch_size": train_config.batch_size,
        "num_workers": train_config.num_workers,
        "pin_memory": train_config.pin_memory,
        "persistent_workers": train_config.persistent_workers,
        "prefetch_factor": train_config.prefetch_factor,
        "cudnn_benchmark": train_config.cudnn_benchmark,
        "channels_last": train_config.channels_last,
        "image_size": model_config.image_size,
        "train_backbone": model_config.backbone.train_backbone,
        "timm_name": model_config.backbone.timm_name,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a short training throughput benchmark.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "bench" / "training_speed.yaml"),
        help="Path to the throughput benchmark YAML.",
    )
    parser.add_argument(
        "--append-jsonl",
        default=None,
        help="Optional JSONL path to append the result row.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    bench_config = load_yaml(config_path)
    result = benchmark_once(bench_config)
    result["config_path"] = str(config_path)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.append_jsonl:
        output_path = Path(args.append_jsonl)
        if not output_path.is_absolute():
            output_path = ROOT / output_path
        append_jsonl(output_path, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
