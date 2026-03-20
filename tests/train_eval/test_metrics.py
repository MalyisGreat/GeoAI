from __future__ import annotations

import torch

from geo_autolab.eval.metrics import build_group_metrics, compute_distance_metrics, expected_calibration_error


def test_compute_distance_metrics_basic_summary() -> None:
    distances = torch.tensor([0.0, 10.0, 110.0], dtype=torch.float32)
    metrics = compute_distance_metrics(distances, thresholds_km=[1, 25, 100])
    assert metrics["mean_km"] > 0.0
    assert abs(metrics["within_1km"] - (1 / 3)) < 1e-6
    assert abs(metrics["within_25km"] - (2 / 3)) < 1e-6


def test_expected_calibration_error_zero_for_perfect_bins() -> None:
    confidence = torch.tensor([1.0, 0.0], dtype=torch.float32)
    correct = torch.tensor([1.0, 0.0], dtype=torch.float32)
    ece = expected_calibration_error(confidence, correct, bins=2)
    assert ece == 0.0


def test_build_group_metrics_tracks_group_accuracy() -> None:
    distances = torch.tensor([10.0, 15.0, 300.0], dtype=torch.float32)
    confidence = torch.tensor([0.9, 0.8, 0.2], dtype=torch.float32)
    correct = torch.tensor([1, 1, 0], dtype=torch.bool)
    groups = ["domain:a", "domain:a", "domain:b"]
    grouped = build_group_metrics(distances, confidence, correct, groups, thresholds_km=[25, 100])
    assert grouped["domain:a"]["within_25km"] == 1.0
    assert grouped["domain:b"]["within_100km"] == 0.0
