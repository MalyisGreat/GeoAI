from __future__ import annotations

from collections import defaultdict

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from geo_autolab.geo import great_circle_distance_km
from geo_autolab.utils import choose_device, to_device

from .config import EvalConfig
from .gates import AntiRewardHackGate
from .metrics import build_group_metrics, compute_distance_metrics, expected_calibration_error


class Evaluator:
    def __init__(self, config: EvalConfig, device: str | None = None) -> None:
        self.config = config
        self.device = choose_device(device)
        self.gate = AntiRewardHackGate(config)

    def evaluate(self, model: nn.Module, loader: DataLoader) -> tuple[dict[str, float], dict[str, dict[str, float]], list[dict[str, float | str]]]:
        model.eval()
        all_distances: list[Tensor] = []
        all_confidence: list[Tensor] = []
        all_correct: list[Tensor] = []
        grouped_labels: list[str] = []
        row_records: list[dict[str, float | str]] = []

        with torch.no_grad():
            for batch in loader:
                batch = to_device(batch, self.device)
                outputs = model(batch["image"])
                pred_xyz = outputs["unit_xyz"]
                target_xyz = batch["unit_xyz"]
                distances = great_circle_distance_km(pred_xyz, target_xyz)
                all_distances.append(distances)

                if outputs["geocell_logits"] is not None:
                    probs = outputs["geocell_logits"].softmax(dim=-1)
                    confidence, predicted = probs.max(dim=-1)
                    correct = predicted.eq(batch["geocell_id"])
                else:
                    confidence = torch.zeros_like(distances)
                    correct = torch.zeros_like(distances, dtype=torch.bool)

                all_confidence.append(confidence)
                all_correct.append(correct)

                domains = batch["domain"]
                sources = batch["source"]
                group_labels = batch.get("group_label", domains)
                for idx in range(len(domains)):
                    grouped_labels.append(str(group_labels[idx]))
                    row_records.append(
                        {
                            "distance_km": float(distances[idx].item()),
                            "confidence": float(confidence[idx].item()),
                            "correct": float(correct[idx].float().item()),
                            "domain": str(domains[idx]),
                            "source": str(sources[idx]),
                            "group_label": str(group_labels[idx]),
                        }
                    )

        distances = torch.cat(all_distances) if all_distances else torch.empty(0)
        confidence = torch.cat(all_confidence) if all_confidence else torch.empty(0)
        correct = torch.cat(all_correct) if all_correct else torch.empty(0, dtype=torch.bool)

        metrics = compute_distance_metrics(distances, self.config.distance_thresholds_km)
        metrics["avg_confidence"] = float(confidence.mean().item()) if confidence.numel() else 0.0
        metrics["geocell_top1"] = float(correct.float().mean().item()) if correct.numel() else 0.0
        metrics["ece"] = expected_calibration_error(confidence, correct) if confidence.numel() else 0.0

        grouped_metrics = build_group_metrics(distances, confidence, correct, grouped_labels, self.config.distance_thresholds_km)
        return metrics, grouped_metrics, row_records

    def evaluate_with_gates(self, model: nn.Module, loader: DataLoader):
        metrics, grouped_metrics, rows = self.evaluate(model, loader)
        report = self.gate.evaluate(metrics, grouped_metrics)
        return report, rows
