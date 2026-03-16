from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from geobot.data import GeoDataset, attach_label_indices, load_frame, split_train_val
from geobot.eval import GalleryIndex, build_cell_state_store, rerank_candidate_cells, summarize_errors
from geobot.model import GeoLocator
from geobot.utils.geo import haversine_km, tensor_unit_to_latlon
from geobot.utils.io import save_manifest
from geobot.utils.logging import init_run_dirs, log_event, save_summary, system_snapshot


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def configure_runtime(device: torch.device, config: dict[str, Any]) -> None:
    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = bool(config["train"].get("cudnn_benchmark", True))
    if bool(config["train"].get("allow_tf32", True)):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision(str(config["train"].get("matmul_precision", "high")))


def _maybe_sample(frame: pd.DataFrame, limit: int | None, seed: int) -> pd.DataFrame:
    if limit is None or len(frame) <= limit:
        return frame.reset_index(drop=True)
    return frame.sample(n=limit, random_state=seed).reset_index(drop=True)


def _prepare_frames(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, Any]:
    train_frame = load_frame(config["data"]["train_manifest"])
    if config["data"].get("val_manifest"):
        val_frame = load_frame(config["data"]["val_manifest"])
    else:
        train_frame, val_frame = split_train_val(
            train_frame,
            val_fraction=float(config["data"].get("val_fraction", 0.1)),
            seed=int(config["seed"]),
        )
    train_frame = _maybe_sample(train_frame, config["data"].get("max_train_rows"), config["seed"])
    val_frame = _maybe_sample(val_frame, config["data"].get("max_val_rows"), config["seed"])
    train_frame, val_frame, label_maps = attach_label_indices(train_frame, val_frame)
    return train_frame, val_frame, label_maps


def _make_loaders(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[GeoDataset, GeoDataset, DataLoader, DataLoader]:
    image_root = config["data"]["image_root"]
    image_size = int(config["model"]["image_size"])
    train_ds = GeoDataset(train_frame, image_root=image_root, image_size=image_size, augment=True)
    val_ds = GeoDataset(val_frame, image_root=image_root, image_size=image_size, augment=False)
    num_workers = int(config["train"].get("num_workers", 0))
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": bool(config["train"].get("pin_memory", False)),
        "persistent_workers": num_workers > 0,
        "prefetch_factor": int(config["train"].get("prefetch_factor", 2)) if num_workers > 0 else None,
    }
    loader_kwargs = {key: value for key, value in loader_kwargs.items() if value is not None}
    train_loader = DataLoader(
        train_ds,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=True,
        drop_last=bool(config["train"].get("drop_last", False)),
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(config["train"]["val_batch_size"]),
        shuffle=False,
        **loader_kwargs,
    )
    return train_ds, val_ds, train_loader, val_loader


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def _probabilistic_loss(outputs: dict[str, torch.Tensor], target_units: torch.Tensor) -> torch.Tensor:
    logits = outputs["posterior_logits"]
    means = outputs["posterior_means"]
    kappa = outputs["posterior_kappa"]
    similarity = (means * target_units.unsqueeze(1)).sum(dim=-1)
    log_mix = F.log_softmax(logits, dim=-1)
    log_prob = torch.logsumexp(log_mix + kappa * similarity, dim=-1)
    return -log_prob.mean()


def _clue_losses(outputs: dict[str, Any], batch: dict[str, Any]) -> torch.Tensor:
    total = None
    for name, logits in outputs.get("clue_logits", {}).items():
        key = f"aux_{name}_idx"
        if key in batch:
            loss = F.cross_entropy(logits, batch[key])
            total = loss if total is None else total + loss
    for name, prediction in outputs.get("clue_numeric", {}).items():
        key = f"aux_{name}_value"
        if key in batch:
            loss = F.mse_loss(prediction, batch[key])
            total = loss if total is None else total + loss
    if total is None:
        return batch["coord_unit"].new_tensor(0.0)
    return total


def _compute_loss(
    outputs: dict[str, torch.Tensor], batch: dict[str, Any], config: dict[str, Any]
) -> tuple[torch.Tensor, dict[str, float]]:
    label_smoothing = float(config["train"].get("label_smoothing", 0.0))
    coarse_loss = F.cross_entropy(
        outputs["coarse_logits"], batch["coarse_idx"], label_smoothing=label_smoothing
    )
    fine_loss = F.cross_entropy(
        outputs["fine_logits"], batch["fine_idx"], label_smoothing=label_smoothing
    )
    retrieval_loss = F.cross_entropy(
        outputs["retrieval_logits"], batch["fine_idx"], label_smoothing=label_smoothing
    )
    regression_loss = (1.0 - (outputs["coord_unit"] * batch["coord_unit"]).sum(dim=-1)).mean()
    distribution_loss = _probabilistic_loss(outputs, batch["coord_unit"])
    clue_loss = _clue_losses(outputs, batch)
    total = (
        float(config["train"].get("coarse_loss_weight", 0.35)) * coarse_loss
        + float(config["train"].get("fine_loss_weight", 0.9)) * fine_loss
        + float(config["train"].get("retrieval_loss_weight", 0.35)) * retrieval_loss
        + float(config["train"].get("regression_loss_weight", 0.65)) * regression_loss
        + float(config["train"].get("distribution_loss_weight", 0.45)) * distribution_loss
        + float(config["train"].get("clue_loss_weight", 0.25)) * clue_loss
    )
    components = {
        "loss": float(total.detach().cpu()),
        "coarse_loss": float(coarse_loss.detach().cpu()),
        "fine_loss": float(fine_loss.detach().cpu()),
        "retrieval_loss": float(retrieval_loss.detach().cpu()),
        "regression_loss": float(regression_loss.detach().cpu()),
        "distribution_loss": float(distribution_loss.detach().cpu()),
        "clue_loss": float(clue_loss.detach().cpu()),
    }
    return total, components


def _predict_units(
    outputs: dict[str, torch.Tensor],
    *,
    gallery: GalleryIndex | None,
    cell_store: Any,
    config: dict[str, Any],
) -> torch.Tensor:
    candidate_units, _ = rerank_candidate_cells(
        outputs,
        cell_store,
        candidate_top_k=int(config["eval"].get("candidate_top_k", config["eval"]["retrieval_top_k"])),
        retrieval_weight=float(config["eval"].get("candidate_retrieval_weight", 0.35)),
        clue_weight=float(config["eval"].get("clue_rerank_weight", 0.2)),
    )
    regressed = outputs["coord_unit"]
    if gallery is not None:
        retrieved = gallery.query(
            outputs.get("retrieval_embedding", outputs["embedding"]),
            regressed,
            outputs["coarse_logits"],
            top_k=int(config["eval"]["retrieval_top_k"]),
            blend=float(config["eval"]["retrieval_blend"]),
            coarse_top_k=int(config["eval"]["retrieval_filter_coarse_top_k"]),
        )
        return torch.nn.functional.normalize(
            0.4 * regressed + 0.25 * candidate_units + 0.35 * retrieved,
            dim=-1,
        )
    return torch.nn.functional.normalize(0.6 * regressed + 0.4 * candidate_units, dim=-1)


def _evaluate(
    model: nn.Module,
    train_ds: GeoDataset,
    val_loader: DataLoader,
    train_frame: pd.DataFrame,
    label_maps: Any,
    *,
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, float]:
    gallery = None
    if bool(config["eval"].get("use_gallery_rerank", True)):
        gallery = GalleryIndex.build(
            model,
            train_ds,
            device=device,
            batch_size=int(config["train"]["val_batch_size"]),
            num_workers=int(config["train"].get("num_workers", 0)),
            max_items=min(len(train_ds), int(config["eval"].get("max_gallery_items", 50000))),
        )

    cell_store = build_cell_state_store(
        train_frame,
        label_maps,
        num_fine_classes=int(train_frame["fine_idx"].max()) + 1,
    )
    coarse_correct = 0
    fine_correct = 0
    total_items = 0
    errors: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            batch = _move_batch(batch, device)
            outputs = model(batch["image"])
            pred_units = _predict_units(outputs, gallery=gallery, cell_store=cell_store, config=config)
            pred_latlon = tensor_unit_to_latlon(pred_units).cpu().numpy()
            true_latlon = batch["latlon"].cpu().numpy()
            batch_errors = haversine_km(
                pred_latlon[:, 0],
                pred_latlon[:, 1],
                true_latlon[:, 0],
                true_latlon[:, 1],
            )
            errors.append(batch_errors)
            coarse_correct += int(
                (outputs["coarse_logits"].argmax(dim=-1) == batch["coarse_idx"]).sum().item()
            )
            fine_correct += int(
                (outputs["fine_logits"].argmax(dim=-1) == batch["fine_idx"]).sum().item()
            )
            total_items += len(batch_errors)

    all_errors = np.concatenate(errors) if errors else np.array([], dtype=np.float32)
    metrics = summarize_errors(all_errors, list(config["eval"]["thresholds_km"]))
    metrics["coarse_top1"] = coarse_correct / max(total_items, 1)
    metrics["fine_top1"] = fine_correct / max(total_items, 1)
    metrics["num_eval_samples"] = float(total_items)
    return metrics


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, float]:
    model.train()
    amp_mode = str(config["train"].get("amp", "none")).lower()
    use_autocast = device.type == "cuda" and amp_mode in {"bf16", "fp16"}
    autocast_dtype = torch.bfloat16 if amp_mode == "bf16" else torch.float16
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler(
            "cuda", enabled=device.type == "cuda" and amp_mode == "fp16"
        )
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and amp_mode == "fp16")
    accumulate_steps = max(1, int(config["train"].get("accumulate_steps", 1)))

    optimizer.zero_grad(set_to_none=True)
    aggregates = {
        "loss": 0.0,
        "coarse_loss": 0.0,
        "fine_loss": 0.0,
        "retrieval_loss": 0.0,
        "regression_loss": 0.0,
        "distribution_loss": 0.0,
        "clue_loss": 0.0,
    }
    total_batches = 0
    total_images = 0
    start_time = time.perf_counter()
    epoch_log_path = Path(config["train"].get("_train_step_log_path", ""))
    last_step_end = start_time

    for step, batch in enumerate(loader, start=1):
        batch_ready_time = time.perf_counter()
        data_wait_seconds = batch_ready_time - last_step_end
        batch = _move_batch(batch, device)
        images = batch["image"]
        if bool(config["train"].get("channels_last", False)) and device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
            batch["image"] = images

        compute_start = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
            outputs = model(batch["image"])
            loss, components = _compute_loss(outputs, batch, config)
            scaled_loss = loss / accumulate_steps

        if scaler.is_enabled():
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        if step % accumulate_steps == 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["train"]["grad_clip_norm"]))
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        for key, value in components.items():
            aggregates[key] += value
        total_batches += 1
        total_images += int(batch["image"].shape[0])
        step_end = time.perf_counter()
        compute_seconds = step_end - compute_start
        total_step_seconds = step_end - batch_ready_time
        if epoch_log_path and step % int(config["train"].get("log_every_steps", 20)) == 0:
            log_event(
                epoch_log_path,
                {
                    "epoch": int(config["train"].get("_current_epoch", 0)),
                    "step": step,
                    "loss": components["loss"],
                    "data_wait_seconds": data_wait_seconds,
                    "compute_seconds": compute_seconds,
                    "step_seconds": total_step_seconds,
                    "step_samples_per_sec": batch["image"].shape[0] / max(total_step_seconds, 1e-6),
                },
            )
        last_step_end = step_end

    duration = max(1e-6, time.perf_counter() - start_time)
    return {
        **{key: value / max(total_batches, 1) for key, value in aggregates.items()},
        "samples_per_sec": total_images / duration,
        "epoch_seconds": duration,
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
    }


def _save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    epoch: int,
    metrics: dict[str, float],
    label_maps: Any,
    cell_centers: torch.Tensor,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config,
        "metrics": metrics,
        "coarse_to_idx": label_maps.coarse_to_idx,
        "fine_to_idx": label_maps.fine_to_idx,
        "aux_categorical": label_maps.aux_categorical,
        "aux_numeric_stats": label_maps.aux_numeric_stats,
        "cell_centers": cell_centers.cpu(),
    }
    torch.save(checkpoint, path)


def run_training(config: dict[str, Any]) -> dict[str, Any]:
    set_seed(int(config["seed"]))
    device = resolve_device(str(config["runtime"].get("device", "auto")))
    configure_runtime(device, config)
    if bool(config["runtime"].get("deterministic", False)):
        torch.use_deterministic_algorithms(True)

    run_dirs = init_run_dirs(config["project_root"], config["experiment_name"])
    save_summary(run_dirs["artifacts"] / "system.json", system_snapshot())

    prepare_started = time.perf_counter()
    train_frame, val_frame, label_maps = _prepare_frames(config)
    save_manifest(train_frame, run_dirs["artifacts"] / "train_manifest.parquet")
    save_manifest(val_frame, run_dirs["artifacts"] / "val_manifest.parquet")
    cell_store = build_cell_state_store(
        train_frame,
        label_maps,
        num_fine_classes=int(train_frame["fine_idx"].max()) + 1,
    )
    prepare_seconds = time.perf_counter() - prepare_started

    loader_started = time.perf_counter()
    train_ds, val_ds, train_loader, val_loader = _make_loaders(train_frame, val_frame, config)
    loader_seconds = time.perf_counter() - loader_started
    model_started = time.perf_counter()
    categorical_heads = config["model"].get("auxiliary_categorical_heads") or list(
        label_maps.aux_categorical.keys()
    )
    numeric_heads = config["model"].get("auxiliary_numeric_heads") or list(
        label_maps.aux_numeric_stats.keys()
    )
    model = GeoLocator(
        backbone_name=config["model"]["backbone"],
        fallback_backbone=config["model"]["fallback_backbone"],
        pretrained=bool(config["model"].get("pretrained", False)),
        hidden_dim=int(config["model"]["hidden_dim"]),
        embedding_dim=int(config["model"]["embedding_dim"]),
        dropout=float(config["model"]["dropout"]),
        num_coarse_classes=len(label_maps.coarse_to_idx),
        num_fine_classes=len(label_maps.fine_to_idx),
        num_experts=int(config["model"].get("num_experts", 4)),
        posterior_components=int(config["model"].get("posterior_components", 8)),
        retrieval_temperature=float(config["model"].get("retrieval_temperature", 0.07)),
        aux_head_dims={
            name: len(label_maps.aux_categorical[name])
            for name in categorical_heads
            if name in label_maps.aux_categorical
        },
        numeric_head_names=[name for name in numeric_heads if name in label_maps.aux_numeric_stats],
    ).to(device)
    model_init_seconds = time.perf_counter() - model_started

    if bool(config["train"].get("compile", False)) and device.type == "cuda" and hasattr(torch, "compile"):
        try:
            model = torch.compile(
                model,
                mode=str(config["train"].get("compile_mode", "reduce-overhead")),
            )  # type: ignore[assignment]
        except Exception:
            pass

    optimizer_kwargs = {
        "lr": float(config["train"]["learning_rate"]),
        "weight_decay": float(config["train"]["weight_decay"]),
    }
    if device.type == "cuda" and bool(config["train"].get("fused_optimizer", True)):
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(config["train"]["epochs"]))
    )

    best_metric = math.inf
    best_checkpoint = run_dirs["checkpoints"] / "best.pt"
    train_log = run_dirs["metrics"] / "train.jsonl"
    val_log = run_dirs["metrics"] / "val.jsonl"
    step_log = run_dirs["metrics"] / "train_steps.jsonl"
    save_summary(
        run_dirs["artifacts"] / "startup.json",
        {
            "prepare_seconds": prepare_seconds,
            "loader_seconds": loader_seconds,
            "model_init_seconds": model_init_seconds,
            "num_train_rows": len(train_frame),
            "num_val_rows": len(val_frame),
        },
    )

    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        config["train"]["_current_epoch"] = epoch
        config["train"]["_train_step_log_path"] = str(step_log)
        train_metrics = _train_one_epoch(model, train_loader, optimizer, device=device, config=config)
        train_metrics["epoch"] = epoch
        log_event(train_log, train_metrics)

        eval_metrics = _evaluate(
            model,
            train_ds,
            val_loader,
            train_frame,
            label_maps,
            device=device,
            config=config,
        )
        eval_metrics["epoch"] = epoch
        log_event(val_log, eval_metrics)
        scheduler.step()

        if eval_metrics["median_km"] < best_metric:
            best_metric = eval_metrics["median_km"]
            _save_checkpoint(
                best_checkpoint,
                model=model,
                optimizer=optimizer,
                config=config,
                epoch=epoch,
                metrics=eval_metrics,
                label_maps=label_maps,
                cell_centers=cell_store.centers,
            )

    summary = {
        "experiment_name": config["experiment_name"],
        "device": str(device),
        "num_train_samples": len(train_ds),
        "num_val_samples": len(val_ds),
        "best_median_km": best_metric,
        "best_checkpoint": str(best_checkpoint),
        "run_root": str(run_dirs["root"]),
        "num_experts": int(config["model"].get("num_experts", 4)),
        "posterior_components": int(config["model"].get("posterior_components", 8)),
        "prepare_seconds": prepare_seconds,
        "loader_seconds": loader_seconds,
        "model_init_seconds": model_init_seconds,
    }
    save_summary(run_dirs["root"] / "summary.json", summary)
    return summary
