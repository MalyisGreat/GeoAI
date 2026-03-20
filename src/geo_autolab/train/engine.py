from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from geo_autolab.config_utils import dump_json
from geo_autolab.utils import choose_device, ensure_dir, seed_everything, to_device

from .config import TrainConfig


@dataclass(slots=True)
class TrainHistory:
    epochs: list[dict[str, object]] = field(default_factory=list)
    best_checkpoint: Path | None = None
    history_path: Path | None = None


def build_optimizer(model: nn.Module, train_config: TrainConfig) -> AdamW:
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
    param_groups = []
    if backbone_params:
        param_groups.append(
            {
                "params": backbone_params,
                "lr": train_config.learning_rate * train_config.backbone_lr_scale,
            }
        )
    if head_params:
        param_groups.append({"params": head_params, "lr": train_config.learning_rate})
    return AdamW(param_groups, weight_decay=train_config.weight_decay)


def build_scheduler(optimizer: AdamW, train_config: TrainConfig, steps_per_epoch: int) -> LambdaLR:
    total_steps = max(1, steps_per_epoch * train_config.max_epochs)
    warmup_steps = max(1, int(total_steps * train_config.warmup_fraction))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535))).item()

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


class Trainer:
    def __init__(self, model: nn.Module, criterion: nn.Module, config: TrainConfig, device: str | None = None) -> None:
        self.model = model
        self.criterion = criterion
        self.config = config
        self.device = choose_device(device)
        self.use_amp = config.amp and self.device.type == "cuda"
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = config.cudnn_benchmark
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp) if self.device.type == "cuda" else None
        self.model.to(self.device)
        if config.channels_last and self.device.type == "cuda":
            self.model.to(memory_format=torch.channels_last)
        self.criterion.to(self.device)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        evaluator,
    ) -> TrainHistory:
        seed_everything(self.config.seed)
        optimizer = build_optimizer(self.model, self.config)
        scheduler = build_scheduler(optimizer, self.config, steps_per_epoch=max(1, len(train_loader)))
        run_dir = ensure_dir(self.config.run_dir)
        history_path = run_dir / "history.json"

        history = TrainHistory()
        history.history_path = history_path
        best_metric = float("inf")
        global_step = 0
        images_seen = 0
        next_eval_at = self.config.eval_every_images if self.config.eval_every_images and self.config.eval_every_images > 0 else None
        gpu_name = None
        gpu_memory_gb = None
        if self.device.type == "cuda":
            props = torch.cuda.get_device_properties(self.device)
            gpu_name = props.name
            gpu_memory_gb = round(props.total_memory / (1024**3), 2)
        print(
            json.dumps(
                {
                    "event": "train_start",
                    "device": str(self.device),
                    "gpu_name": gpu_name,
                    "gpu_memory_gb": gpu_memory_gb,
                    "amp": self.use_amp,
                    "batch_size": self.config.batch_size,
                    "eval_batch_size": self.config.eval_batch_size,
                    "grad_accum_steps": self.config.grad_accum_steps,
                    "max_epochs": self.config.max_epochs,
                    "run_dir": str(run_dir),
                    "manifest_path": self.config.manifest_path,
                    "init_checkpoint": self.config.init_checkpoint,
                },
                sort_keys=True,
            ),
            flush=True,
        )

        for epoch in range(self.config.max_epochs):
            print(
                json.dumps(
                    {
                        "event": "epoch_start",
                        "epoch": epoch,
                        "images_seen": images_seen,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            train_metrics, best_metric, images_seen, next_eval_at = self._train_epoch(
                train_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                evaluator=evaluator,
                val_loader=val_loader,
                history=history,
                history_path=history_path,
                run_dir=run_dir,
                best_metric=best_metric,
                images_seen=images_seen,
                next_eval_at=next_eval_at,
            )
            global_step += len(train_loader)
            report, _rows = evaluator.evaluate_with_gates(self.model, val_loader)
            best_metric = self._record_eval(
                history=history,
                history_path=history_path,
                run_dir=run_dir,
                report=report,
                epoch=epoch,
                images_seen=images_seen,
                train_metrics=train_metrics,
                best_metric=best_metric,
                event="epoch_end",
            )
        return history

    def _record_eval(
        self,
        history: TrainHistory,
        history_path: Path,
        run_dir: Path,
        report,
        epoch: int,
        images_seen: int,
        train_metrics: dict[str, float],
        best_metric: float,
        event: str,
        step: int | None = None,
    ) -> float:
        row: dict[str, object] = {
            "event": event,
            "epoch": float(epoch),
            "images_seen": float(images_seen),
            "train_total_loss": train_metrics["total"],
            "train_country_loss": train_metrics["country"],
            "train_region_loss": train_metrics["region"],
            "train_geocell_loss": train_metrics["geocell"],
            "train_geodesic_loss": train_metrics["geodesic"],
            "train_embedding_loss": train_metrics["embedding"],
            "train_offset_loss": train_metrics["offset"],
            "train_hierarchy_loss": train_metrics["hierarchy"],
            "val_median_km": report.metrics["median_km"],
            "val_within_100km": report.metrics.get("within_100km", 0.0),
            "accepted": float(report.accepted),
        }
        if step is not None:
            row["step"] = float(step)
        history.epochs.append(row)
        print(
            json.dumps(
                {
                    "event": event,
                    "epoch": epoch,
                    "images_seen": images_seen,
                    "step": step,
                    "median_km": report.metrics["median_km"],
                    "within_100km": report.metrics.get("within_100km", 0.0),
                    "accepted": report.accepted,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if report.primary_metric < best_metric:
            best_metric = report.primary_metric
            checkpoint_path = run_dir / "best.pt"
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "epoch": epoch,
                    "report": report.metrics,
                },
                checkpoint_path,
            )
            history.best_checkpoint = checkpoint_path
        dump_json(
            history_path,
            {
                "epochs": history.epochs,
                "best_metric": None if best_metric == float("inf") else best_metric,
                "best_checkpoint": str(history.best_checkpoint) if history.best_checkpoint is not None else None,
            },
        )
        return best_metric

    def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: AdamW,
        scheduler: LambdaLR,
        epoch: int,
        global_step: int,
        evaluator,
        val_loader: DataLoader,
        history: TrainHistory,
        history_path: Path,
        run_dir: Path,
        best_metric: float,
        images_seen: int,
        next_eval_at: int | None,
    ) -> tuple[dict[str, float], float, int, int | None]:
        del global_step
        self.model.train()
        running = {
            "total": 0.0,
            "country": 0.0,
            "region": 0.0,
            "geocell": 0.0,
            "geodesic": 0.0,
            "embedding": 0.0,
            "offset": 0.0,
            "hierarchy": 0.0,
        }
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, start=1):
            batch = to_device(batch, self.device)
            if self.config.channels_last and isinstance(batch.get("image"), torch.Tensor):
                batch["image"] = batch["image"].contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(batch["image"])
                positive_outputs = None
                if isinstance(batch.get("positive_image"), torch.Tensor):
                    positive_batch = batch["positive_image"]
                    if self.config.channels_last and isinstance(positive_batch, torch.Tensor):
                        positive_batch = positive_batch.contiguous(memory_format=torch.channels_last)
                    positive_outputs = self.model(positive_batch)
                losses = self.criterion(outputs, batch, positive_outputs=positive_outputs)
                loss = losses.total / self.config.grad_accum_steps

            if self.use_amp:
                assert self.scaler is not None
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % self.config.grad_accum_steps == 0:
                if self.use_amp:
                    assert self.scaler is not None
                    self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                if self.use_amp:
                    assert self.scaler is not None
                    scale_before = self.scaler.get_scale()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    scale_after = self.scaler.get_scale()
                else:
                    optimizer.step()
                    scale_before = None
                    scale_after = None
                optimizer.zero_grad(set_to_none=True)
                if (not self.use_amp) or (scale_before is not None and scale_after is not None and scale_after >= scale_before):
                    scheduler.step()

            running["total"] += float(losses.total.item())
            running["country"] += float(losses.country.item())
            running["region"] += float(losses.region.item())
            running["geocell"] += float(losses.geocell.item())
            running["geodesic"] += float(losses.geodesic.item())
            running["embedding"] += float(losses.embedding.item())
            running["offset"] += float(losses.offset.item())
            running["hierarchy"] += float(losses.hierarchy.item())
            images_seen += int(batch["image"].shape[0])

            if self.config.log_every > 0 and step % self.config.log_every == 0:
                lrs = [float(group["lr"]) for group in optimizer.param_groups]
                denom_so_far = max(1, step)
                print(
                    json.dumps(
                        {
                            "event": "train_step",
                            "epoch": epoch,
                            "step": step,
                            "images_seen": images_seen,
                            "lr_backbone": lrs[0],
                            "lr_head": lrs[-1],
                            "train_total_loss": running["total"] / denom_so_far,
                            "train_country_loss": running["country"] / denom_so_far,
                            "train_region_loss": running["region"] / denom_so_far,
                            "train_geocell_loss": running["geocell"] / denom_so_far,
                            "train_geodesic_loss": running["geodesic"] / denom_so_far,
                            "train_embedding_loss": running["embedding"] / denom_so_far,
                            "train_offset_loss": running["offset"] / denom_so_far,
                            "train_hierarchy_loss": running["hierarchy"] / denom_so_far,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

            if next_eval_at is not None and images_seen >= next_eval_at:
                denom_so_far = max(1, step)
                partial_metrics = {key: value / denom_so_far for key, value in running.items()}
                report, _rows = evaluator.evaluate_with_gates(self.model, val_loader)
                best_metric = self._record_eval(
                    history=history,
                    history_path=history_path,
                    run_dir=run_dir,
                    report=report,
                    epoch=epoch,
                    images_seen=images_seen,
                    train_metrics=partial_metrics,
                    best_metric=best_metric,
                    event="periodic",
                    step=step,
                )
                while next_eval_at is not None and images_seen >= next_eval_at:
                    next_eval_at += self.config.eval_every_images or 0

        denom = max(1, len(train_loader))
        return {key: value / denom for key, value in running.items()}, best_metric, images_seen, next_eval_at
