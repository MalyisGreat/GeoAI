from __future__ import annotations

from geo_autolab.config import EvalConfig
from geo_autolab.results import ExperimentResult


def diagnose_result(result: ExperimentResult, cfg: EvalConfig) -> list[str]:
    notes = " ".join(result.notes).lower()
    diagnostics: list[str] = []
    if "oom" in notes:
        diagnostics.append("oom")

    primary = cfg.primary_metric
    train_primary = result.train_metrics.get(primary)
    val_primary = result.split_metrics.get("val", {}).get(primary)
    if train_primary is not None and val_primary is not None and primary.endswith("_km"):
        gap = (val_primary - train_primary) / max(val_primary, 1e-6)
        if gap > cfg.max_train_val_gap:
            diagnostics.append("overfit")

    val_conf = result.split_metrics.get("val", {}).get("confidence_error", 0.0)
    if val_conf > cfg.max_confidence_error:
        diagnostics.append("overfit")

    stress = result.split_metrics.get("stress", {}).get(primary)
    if val_primary is not None and stress is not None and primary.endswith("_km") and stress > val_primary * 1.1:
        diagnostics.append("weak-stress")

    if result.status == "completed" and val_primary is not None and val_primary > 2500:
        diagnostics.append("stalled")

    return diagnostics
