from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from geo_autolab.autonomy.config import AutoLabConfig
from geo_autolab.autonomy.runner import LocalExperimentExecutor
from geo_autolab.config import load_model
from geo_autolab.contracts import ExperimentSpec
from geo_autolab.models import ModelConfig
from geo_autolab.utils import append_jsonl, ensure_dir


@dataclass(slots=True)
class SweepVariant:
    name: str
    timm_name: str
    image_size: int
    batch_size: int
    eval_batch_size: int
    max_epochs: int
    learning_rate: float
    train_backbone: bool = False
    backbone_lr_scale: float = 0.0
    checkpoint_gradients: bool = False
    grad_accum_steps: int | None = None
    weight_decay: float | None = None
    warmup_fraction: float | None = None
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    num_workers: int | None = None
    init_checkpoint: str | None = None
    head_hidden_dim: int | None = None
    head_embedding_dim: int | None = None
    head_dropout: float | None = None
    adapter_enabled: bool | None = None
    adapter_bottleneck_dim: int | None = None
    adapter_dropout: float | None = None


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def default_variants() -> list[SweepVariant]:
    return [
        SweepVariant(
            name="convnextv2-tiny-ft",
            timm_name="convnextv2_tiny",
            image_size=192,
            batch_size=16,
            eval_batch_size=32,
            max_epochs=2,
            learning_rate=0.0004,
            train_backbone=True,
            backbone_lr_scale=0.1,
            max_train_samples=10000,
            max_val_samples=2000,
            num_workers=6,
        ),
        SweepVariant(
            name="convnext-tiny-ft",
            timm_name="convnext_tiny",
            image_size=192,
            batch_size=16,
            eval_batch_size=32,
            max_epochs=2,
            learning_rate=0.0004,
            train_backbone=True,
            backbone_lr_scale=0.1,
            max_train_samples=10000,
            max_val_samples=2000,
            num_workers=6,
        ),
        SweepVariant(
            name="tf-efficientnetv2-b0-ft",
            timm_name="tf_efficientnetv2_b0",
            image_size=192,
            batch_size=24,
            eval_batch_size=48,
            max_epochs=2,
            learning_rate=0.0005,
            train_backbone=True,
            backbone_lr_scale=0.1,
            max_train_samples=10000,
            max_val_samples=2000,
            num_workers=6,
        ),
        SweepVariant(
            name="efficientnet-b2-ft",
            timm_name="efficientnet_b2",
            image_size=192,
            batch_size=20,
            eval_batch_size=40,
            max_epochs=2,
            learning_rate=0.00045,
            train_backbone=True,
            backbone_lr_scale=0.08,
            max_train_samples=10000,
            max_val_samples=2000,
            num_workers=6,
        ),
        SweepVariant(
            name="mobilenetv3-large-ft",
            timm_name="mobilenetv3_large_100",
            image_size=192,
            batch_size=32,
            eval_batch_size=64,
            max_epochs=2,
            learning_rate=0.0005,
            train_backbone=True,
            backbone_lr_scale=0.1,
            max_train_samples=10000,
            max_val_samples=2000,
            num_workers=6,
        ),
        SweepVariant(
            name="vit-small-ft",
            timm_name="vit_small_patch16_224",
            image_size=224,
            batch_size=8,
            eval_batch_size=16,
            max_epochs=2,
            learning_rate=0.0003,
            train_backbone=True,
            backbone_lr_scale=0.05,
            max_train_samples=10000,
            max_val_samples=2000,
            num_workers=6,
        ),
    ]


def parse_variant_payload(raw: str) -> SweepVariant:
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise TypeError("Each variant payload must decode to an object.")
    return SweepVariant(**payload)


def load_variants(args: argparse.Namespace) -> list[SweepVariant]:
    variants: list[SweepVariant] = []
    if args.variant:
        variants.extend(parse_variant_payload(raw) for raw in args.variant)
    if args.variant_file:
        payload = json.loads(Path(args.variant_file).read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise TypeError("Variant file must contain a JSON array.")
        variants.extend(SweepVariant(**item) for item in payload)
    if variants:
        return variants
    return default_variants()


def build_variant_spec(base_config: AutoLabConfig, variant: SweepVariant, run_root: Path) -> tuple[AutoLabConfig, ExperimentSpec]:
    config = base_config.model_copy(deep=True)
    model_config = load_model(config.model_config_path, ModelConfig)
    model_payload = model_config.model_copy(deep=True)

    model_payload.backbone.timm_name = variant.timm_name
    model_payload.backbone.train_backbone = variant.train_backbone
    model_payload.backbone.checkpoint_gradients = variant.checkpoint_gradients
    model_payload.image_size = variant.image_size
    if variant.head_hidden_dim is not None:
        model_payload.head.hidden_dim = variant.head_hidden_dim
    if variant.head_embedding_dim is not None:
        model_payload.head.embedding_dim = variant.head_embedding_dim
    if variant.head_dropout is not None:
        model_payload.head.dropout = variant.head_dropout
    if variant.adapter_enabled is not None:
        model_payload.adapter.enabled = variant.adapter_enabled
    if variant.adapter_bottleneck_dim is not None:
        model_payload.adapter.bottleneck_dim = variant.adapter_bottleneck_dim
    if variant.adapter_dropout is not None:
        model_payload.adapter.dropout = variant.adapter_dropout

    config.name = f"arch-sweep-{variant.name}"
    config.run_root = str(run_root)
    config.history_path = str(run_root / "history.jsonl")
    config.train.batch_size = variant.batch_size
    config.train.eval_batch_size = variant.eval_batch_size
    config.train.max_epochs = variant.max_epochs
    config.train.learning_rate = variant.learning_rate
    config.train.backbone_lr_scale = variant.backbone_lr_scale
    config.train.init_checkpoint = None
    if variant.grad_accum_steps is not None:
        config.train.grad_accum_steps = variant.grad_accum_steps
    if variant.weight_decay is not None:
        config.train.weight_decay = variant.weight_decay
    if variant.warmup_fraction is not None:
        config.train.warmup_fraction = variant.warmup_fraction
    if variant.max_train_samples is not None:
        config.train.max_train_samples = variant.max_train_samples
    if variant.max_val_samples is not None:
        config.train.max_val_samples = variant.max_val_samples
    if variant.num_workers is not None:
        config.train.num_workers = variant.num_workers
    if variant.init_checkpoint is not None:
        config.train.init_checkpoint = variant.init_checkpoint

    spec = ExperimentSpec(
        name=_slug(variant.name),
        cycle_index=0,
        model=model_payload.model_dump(),
        train=config.train.model_dump(),
        evaluation=config.evaluation.model_dump(),
        notes=[
            f"architecture sweep variant={variant.name}",
            f"timm_name={variant.timm_name}",
            f"image_size={variant.image_size}",
        ],
    )
    return config, spec


def load_history_summary(checkpoint_path: str | None) -> dict[str, Any]:
    if checkpoint_path is None:
        return {}
    history_path = Path(checkpoint_path).with_name("history.json")
    if not history_path.exists():
        return {}
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    epochs = payload.get("epochs", [])
    if not epochs:
        return {"history_path": str(history_path), "epochs_ran": 0}

    first = epochs[0]
    last = epochs[-1]
    first_median = first.get("val_median_km")
    last_median = last.get("val_median_km")
    first_within_100 = first.get("val_within_100km")
    last_within_100 = last.get("val_within_100km")
    improvement_km = None
    if first_median is not None and last_median is not None:
        improvement_km = float(first_median) - float(last_median)
    improvement_within_100 = None
    if first_within_100 is not None and last_within_100 is not None:
        improvement_within_100 = float(last_within_100) - float(first_within_100)
    return {
        "history_path": str(history_path),
        "epochs_ran": len(epochs),
        "first_val_median_km": first_median,
        "last_val_median_km": last_median,
        "median_improvement_km": improvement_km,
        "first_val_within_100km": first_within_100,
        "last_val_within_100km": last_within_100,
        "within_100km_improvement": improvement_within_100,
        "epochs": epochs,
    }


def summarize_result(variant: SweepVariant, result) -> dict[str, Any]:
    metrics = result.report.metrics
    row = {
        "name": variant.name,
        "timm_name": variant.timm_name,
        "image_size": variant.image_size,
        "batch_size": variant.batch_size,
        "eval_batch_size": variant.eval_batch_size,
        "max_epochs": variant.max_epochs,
        "learning_rate": variant.learning_rate,
        "train_backbone": variant.train_backbone,
        "backbone_lr_scale": variant.backbone_lr_scale,
        "grad_accum_steps": variant.grad_accum_steps,
        "weight_decay": variant.weight_decay,
        "warmup_fraction": variant.warmup_fraction,
        "max_train_samples": variant.max_train_samples,
        "max_val_samples": variant.max_val_samples,
        "num_workers": variant.num_workers,
        "init_checkpoint": variant.init_checkpoint,
        "accepted": result.report.accepted,
        "primary_metric": result.report.primary_metric,
        "median_km": metrics.get("median_km"),
        "within_100km": metrics.get("within_100km"),
        "within_750km": metrics.get("within_750km"),
        "within_2500km": metrics.get("within_2500km"),
        "geocell_top1": metrics.get("geocell_top1"),
        "ece": metrics.get("ece"),
        "flags": result.report.suspicious_flags,
        "checkpoint_path": str(result.checkpoint_path) if result.checkpoint_path else None,
    }
    row.update(load_history_summary(row["checkpoint_path"]))
    return row


def print_ranked_table(rows: list[dict[str, Any]]) -> None:
    ranked = sorted(
        rows,
        key=lambda row: (
            float("inf") if row.get("median_km") is None else row["median_km"],
            -(row.get("within_100km") or 0.0),
        ),
    )
    headers = ("rank", "name", "median_km", "within_100km", "improve_km", "improve_100", "accepted")
    table_rows = []
    for index, row in enumerate(ranked, start=1):
        table_rows.append(
            (
                str(index),
                str(row["name"]),
                f"{(row.get('median_km') or 0.0):.2f}",
                f"{(row.get('within_100km') or 0.0):.4f}",
                f"{(row.get('median_improvement_km') or 0.0):.2f}",
                f"{(row.get('within_100km_improvement') or 0.0):.4f}",
                str(bool(row.get("accepted"))),
            )
        )

    widths = [len(header) for header in headers]
    for row in table_rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]

    def format_row(values: tuple[str, ...]) -> str:
        return "  ".join(value.ljust(width) for value, width in zip(values, widths))

    print(format_row(headers))
    print(format_row(tuple("-" * width for width in widths)))
    for row in table_rows:
        print(format_row(row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a sequential architecture sweep on the Geo AutoLab stack.")
    parser.add_argument(
        "--base-config",
        default=str(ROOT / "configs" / "autonomy" / "osv5m_benchmark_10k.yaml"),
        help="Base AutoLab YAML used for train/eval settings.",
    )
    parser.add_argument(
        "--run-root",
        default=str(ROOT / "runs" / "architecture_sweep"),
        help="Directory where checkpoints and summary files will be written.",
    )
    parser.add_argument(
        "--summary-path",
        default=None,
        help="Optional JSONL output path. Defaults to <run-root>/results.jsonl.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="Inline JSON object describing one variant. Can be repeated.",
    )
    parser.add_argument(
        "--variant-file",
        default=None,
        help="Path to a JSON file containing a list of variant objects.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on how many variants to run after loading presets/custom variants.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    variants = load_variants(args)
    if args.limit is not None:
        variants = variants[: args.limit]
    if not variants:
        raise RuntimeError("No architecture variants were selected.")

    run_root = Path(args.run_root)
    ensure_dir(run_root)
    summary_path = Path(args.summary_path) if args.summary_path else run_root / "results.jsonl"
    base_config = load_model(args.base_config, AutoLabConfig)

    rows: list[dict[str, Any]] = []
    for variant in variants:
        variant_config, spec = build_variant_spec(base_config, variant, run_root=run_root)
        executor = LocalExperimentExecutor(variant_config)
        try:
            result = executor.run(spec)
            row = summarize_result(variant, result)
            row["status"] = "completed"
        except Exception as exc:  # noqa: BLE001 - sweep should continue and record failures
            row = {
                "name": variant.name,
                "timm_name": variant.timm_name,
                "image_size": variant.image_size,
                "batch_size": variant.batch_size,
                "eval_batch_size": variant.eval_batch_size,
                "max_epochs": variant.max_epochs,
                "learning_rate": variant.learning_rate,
                "train_backbone": variant.train_backbone,
                "backbone_lr_scale": variant.backbone_lr_scale,
                "grad_accum_steps": variant.grad_accum_steps,
                "weight_decay": variant.weight_decay,
                "warmup_fraction": variant.warmup_fraction,
                "max_train_samples": variant.max_train_samples,
                "max_val_samples": variant.max_val_samples,
                "num_workers": variant.num_workers,
                "init_checkpoint": variant.init_checkpoint,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        append_jsonl(summary_path, row)
        rows.append(row)
        print(json.dumps(row, indent=2))

    completed = [row for row in rows if row.get("status") == "completed"]
    if completed:
        print()
        print_ranked_table(completed)
    else:
        print("No completed runs to rank.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
