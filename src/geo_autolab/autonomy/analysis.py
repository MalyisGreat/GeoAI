from __future__ import annotations

from typing import Any


def analyze_candidate(
    reference_metrics: dict[str, Any] | None,
    candidate_metrics: dict[str, Any],
) -> dict[str, Any]:
    if reference_metrics is None:
        return {
            "delta_geocell_top1": candidate_metrics.get("geocell_top1", 0.0),
            "delta_country_top1": candidate_metrics.get("country_top1", 0.0),
            "delta_mean_geodesic_km": -candidate_metrics.get("mean_geodesic_km", 0.0),
            "stress_drop": candidate_metrics.get("stress_drop", 0.0),
            "shortcut_risk": candidate_metrics.get("shortcut_risk", 0.0),
            "status": "baseline",
        }

    delta_geocell = candidate_metrics.get("geocell_top1", 0.0) - reference_metrics.get("geocell_top1", 0.0)
    delta_country = candidate_metrics.get("country_top1", 0.0) - reference_metrics.get("country_top1", 0.0)
    delta_geo_km = reference_metrics.get("mean_geodesic_km", 0.0) - candidate_metrics.get(
        "mean_geodesic_km", 0.0
    )
    calibration_regression = candidate_metrics.get("calibration_ece", 0.0) - reference_metrics.get(
        "calibration_ece", 0.0
    )
    stress_regression = candidate_metrics.get("stress_drop", 0.0) - reference_metrics.get(
        "stress_drop", 0.0
    )
    status = "improved"
    if delta_geocell < 0 and delta_geo_km < 0:
        status = "regressed"
    elif delta_geocell > 0 and (stress_regression > 0.04 or calibration_regression > 0.03):
        status = "shortcut-risk"

    return {
        "delta_geocell_top1": delta_geocell,
        "delta_country_top1": delta_country,
        "delta_mean_geodesic_km": delta_geo_km,
        "delta_calibration_ece": calibration_regression,
        "delta_stress_drop": stress_regression,
        "stress_drop": candidate_metrics.get("stress_drop", 0.0),
        "shortcut_risk": candidate_metrics.get("shortcut_risk", 0.0),
        "status": status,
    }
