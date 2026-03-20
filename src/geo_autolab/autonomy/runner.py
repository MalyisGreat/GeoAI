from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from geo_autolab.config import load_model, resolve_path
from geo_autolab.contracts import ExperimentResult, ExperimentSpec
from geo_autolab.eval import EvalConfig, Evaluator
from geo_autolab.models import ModelConfig, build_model_stack
from geo_autolab.train import TrainConfig, Trainer
from geo_autolab.train.dataset import build_dataloaders
from geo_autolab.train.geocells import compute_geocell_centroids, compute_hierarchy_info, infer_geocell_classes
from geo_autolab.utils import ensure_dir

from .config import AutoLabConfig

def load_initial_checkpoint(model: nn.Module, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    source_state = checkpoint["model_state_dict"]
    target_state = model.state_dict()
    merged_state = dict(target_state)

    for key, value in source_state.items():
        if key not in target_state:
            continue
        target_value = target_state[key]
        if target_value.shape == value.shape:
            merged_state[key] = value
            continue

        # Preserve old class rows when classification heads grow.
        if key in {
            "head.country_head.weight",
            "head.country_head.bias",
            "head.region_head.weight",
            "head.region_head.bias",
            "head.geocell_head.weight",
            "head.geocell_head.bias",
            "head.country_embeddings",
            "head.region_embeddings",
        }:
            slices = tuple(slice(0, min(a, b)) for a, b in zip(value.shape, target_value.shape))
            target_clone = target_value.clone()
            target_clone[slices] = value[slices]
            merged_state[key] = target_clone
            continue

    model.load_state_dict(merged_state, strict=False)


def build_initial_spec(config: AutoLabConfig) -> ExperimentSpec:
    model_config = load_model(config.model_config_path, ModelConfig).model_dump()
    return ExperimentSpec(
        name="cycle-00-bootstrap",
        cycle_index=0,
        model=model_config,
        train=config.train.model_dump(),
        evaluation=config.evaluation.model_dump(),
        notes=["Bootstrap candidate generated from the local baseline config."],
    )


class LocalExperimentExecutor:
    def __init__(self, config: AutoLabConfig) -> None:
        self.config = config

    def run(self, spec: ExperimentSpec) -> ExperimentResult:
        model_config = ModelConfig.model_validate(spec.model)
        train_config = TrainConfig.model_validate(spec.train)
        eval_config = EvalConfig.model_validate(spec.evaluation)
        manifest_path = resolve_path(train_config.manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}. "
                "Create a CSV or JSONL with train/val rows before running training."
            )
        train_config.manifest_path = str(manifest_path)
        if model_config.head.geocell_classes <= 0:
            model_config.head.geocell_classes = infer_geocell_classes(manifest_path)
        hierarchy_info = compute_hierarchy_info(manifest_path, geocell_classes=model_config.head.geocell_classes)
        model_config.head.country_classes = len(hierarchy_info.country_labels)
        model_config.head.region_classes = len(hierarchy_info.region_labels)
        geocell_centroids = compute_geocell_centroids(manifest_path, geocell_classes=model_config.head.geocell_classes)

        run_dir = resolve_path(self.config.run_root) / spec.name
        ensure_dir(run_dir)
        train_config.run_dir = str(run_dir)

        model, criterion = build_model_stack(model_config)
        model.set_geocell_centroids(geocell_centroids)
        criterion.set_geocell_centroids(geocell_centroids)
        criterion.set_hierarchy(
            geocell_to_country=hierarchy_info.geocell_to_country,
            geocell_to_region=hierarchy_info.geocell_to_region,
            country_classes=model_config.head.country_classes,
            region_classes=model_config.head.region_classes,
        )
        if train_config.init_checkpoint:
            load_initial_checkpoint(model, resolve_path(train_config.init_checkpoint))
        train_loader, val_loader = build_dataloaders(model_config.image_size, train_config, hierarchy_info=hierarchy_info)
        evaluator = Evaluator(eval_config)
        trainer = Trainer(model, criterion, train_config)
        history = trainer.fit(train_loader, val_loader, evaluator)

        if history.best_checkpoint is not None:
            checkpoint = torch.load(history.best_checkpoint, map_location="cpu")
            model.load_state_dict(checkpoint["model_state_dict"])
        report, _rows = evaluator.evaluate_with_gates(model, val_loader)
        return ExperimentResult(spec=spec, report=report, checkpoint_path=history.best_checkpoint)
