from __future__ import annotations

from collections import defaultdict

import torch
from torch import Tensor


def compute_distance_metrics(distances_km: Tensor, thresholds_km: list[int]) -> dict[str, float]:
    total = max(1, int(distances_km.numel()))
    metrics = {
        "mean_km": float(distances_km.mean().item()),
        "median_km": float(distances_km.median().item()),
        "p90_km": float(torch.quantile(distances_km, 0.9).item()),
    }
    for threshold in thresholds_km:
        hits = int((distances_km <= threshold).sum().item())
        metrics[f"within_{threshold}km"] = hits / total
    return metrics


def expected_calibration_error(confidence: Tensor, correct: Tensor, bins: int = 10) -> float:
    if confidence.numel() == 0:
        return 0.0
    boundaries = torch.linspace(0.0, 1.0, steps=bins + 1, device=confidence.device)
    total = torch.zeros((), device=confidence.device)
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        mask = (confidence >= lower) & (confidence < upper)
        if not mask.any():
            continue
        acc = correct[mask].float().mean()
        conf = confidence[mask].mean()
        total += (mask.float().mean()) * torch.abs(acc - conf)
    return float(total.item())


def build_group_metrics(
    distances_km: Tensor,
    confidence: Tensor,
    correct: Tensor,
    groups: list[str],
    thresholds_km: list[int],
) -> dict[str, dict[str, float]]:
    group_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        group_to_indices[group].append(index)

    summary: dict[str, dict[str, float]] = {}
    for group, indices in group_to_indices.items():
        idx = torch.tensor(indices, device=distances_km.device, dtype=torch.long)
        summary[group] = compute_distance_metrics(distances_km[idx], thresholds_km)
        if confidence.numel():
            summary[group]["avg_confidence"] = float(confidence[idx].mean().item())
            summary[group]["accuracy"] = float(correct[idx].float().mean().item())
    return summary
