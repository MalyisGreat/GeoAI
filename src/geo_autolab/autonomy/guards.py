from __future__ import annotations

from .schemas import GuardConfig, PromotionDecision


def evaluate_promotion(
    reference_metrics: dict[str, float] | None,
    candidate_metrics: dict[str, float],
    guard: GuardConfig,
) -> PromotionDecision:
    if reference_metrics is None:
        reasons = ["accepted initial baseline"]
        score = candidate_metrics.get("geocell_top1", 0.0) - candidate_metrics.get("shortcut_risk", 0.0)
        return PromotionDecision(promote=True, score=score, reasons=reasons)

    reasons: list[str] = []
    promote = True
    delta_geocell = candidate_metrics.get("geocell_top1", 0.0) - reference_metrics.get("geocell_top1", 0.0)
    delta_country = candidate_metrics.get("country_top1", 0.0) - reference_metrics.get("country_top1", 0.0)
    delta_geodesic = reference_metrics.get("mean_geodesic_km", 0.0) - candidate_metrics.get(
        "mean_geodesic_km", 0.0
    )
    if delta_geocell < guard.min_geocell_gain and delta_geodesic < guard.min_geodesic_improvement_km:
        promote = False
        reasons.append("insufficient primary metric improvement")
    if delta_country < -guard.max_country_regression:
        promote = False
        reasons.append("country accuracy regressed too far")
    if candidate_metrics.get("calibration_ece", 0.0) > guard.max_calibration_ece:
        promote = False
        reasons.append("calibration exceeded safe ceiling")
    if candidate_metrics.get("stress_drop", 0.0) > guard.max_stress_drop:
        promote = False
        reasons.append("augmentation stress drop too large")
    if candidate_metrics.get("shortcut_risk", 0.0) > guard.max_shortcut_risk:
        promote = False
        reasons.append("shortcut risk too high")

    reward_hack_pattern = delta_geocell > 0 and (
        candidate_metrics.get("stress_drop", 0.0) > reference_metrics.get("stress_drop", 0.0) + 0.03
        or candidate_metrics.get("calibration_ece", 0.0)
        > reference_metrics.get("calibration_ece", 0.0) + 0.03
    )
    if reward_hack_pattern:
        promote = False
        reasons.append("reward-hacking guard triggered")

    if promote:
        reasons.append("passed improvement and integrity gates")

    score = (
        candidate_metrics.get("geocell_top1", 0.0) * 0.45
        + candidate_metrics.get("country_top1", 0.0) * 0.20
        - candidate_metrics.get("mean_geodesic_km", 0.0) * 0.0005
        - candidate_metrics.get("calibration_ece", 0.0) * 0.15
        - candidate_metrics.get("shortcut_risk", 0.0) * 0.20
    )
    return PromotionDecision(promote=promote, score=score, reasons=reasons)
