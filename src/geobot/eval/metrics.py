from __future__ import annotations

import numpy as np


def summarize_errors(errors_km: np.ndarray, thresholds_km: list[int | float]) -> dict[str, float]:
    metrics = {
        "mean_km": float(np.mean(errors_km)) if len(errors_km) else 0.0,
        "median_km": float(np.median(errors_km)) if len(errors_km) else 0.0,
        "p90_km": float(np.percentile(errors_km, 90)) if len(errors_km) else 0.0,
        "max_km": float(np.max(errors_km)) if len(errors_km) else 0.0,
    }
    for threshold in thresholds_km:
        metrics[f"within_{int(threshold)}km"] = float(np.mean(errors_km <= threshold))
    return metrics
