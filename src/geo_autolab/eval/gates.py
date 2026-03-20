from __future__ import annotations

from geo_autolab.contracts import EvalReport

from .config import EvalConfig


class AntiRewardHackGate:
    def __init__(self, config: EvalConfig) -> None:
        self.config = config

    def evaluate(
        self,
        metrics: dict[str, float],
        grouped_metrics: dict[str, dict[str, float]],
    ) -> EvalReport:
        suspicious_flags: list[str] = []
        notes: list[str] = []

        primary_metric = metrics["median_km"]
        within_100 = metrics.get("within_100km", 0.0)
        geocell_top1 = metrics.get("geocell_top1", 0.0)
        confidence_gap = metrics.get("avg_confidence", 0.0) - metrics.get("geocell_top1", 0.0)

        if primary_metric > self.config.max_primary_metric_km:
            suspicious_flags.append("median-distance-too-high")
        if within_100 < self.config.min_within_100km:
            suspicious_flags.append("weak-local-accuracy")
        if metrics.get("ece", 0.0) > self.config.max_ece:
            suspicious_flags.append("overconfident-calibration")
        if confidence_gap > self.config.max_confidence_gap:
            suspicious_flags.append("confidence-outpaces-accuracy")
        if geocell_top1 - within_100 > self.config.suspicious_geocell_gap:
            suspicious_flags.append("coarse-cell-shortcut-risk")

        worst_group_within = 1.0
        group_within_scores: list[float] = []
        for group_name, group_metrics in grouped_metrics.items():
            if "within_100km" not in group_metrics:
                continue
            group_within = group_metrics["within_100km"]
            group_within_scores.append(group_within)
            worst_group_within = min(worst_group_within, group_within)
            if group_metrics["within_100km"] < self.config.min_worst_group_within_100km:
                suspicious_flags.append(f"group-collapse:{group_name}")
        if group_within_scores:
            best_group = max(group_within_scores)
            if best_group > 0 and worst_group_within > 0:
                gap_ratio = best_group / worst_group_within
                metrics["group_gap_ratio"] = gap_ratio
                if gap_ratio > self.config.max_group_gap_ratio:
                    suspicious_flags.append("group-gap-too-wide")
            else:
                metrics["group_gap_ratio"] = float("inf")
                suspicious_flags.append("group-collapse-zero")

        accepted = not suspicious_flags
        if accepted:
            notes.append("Evaluation gates accepted this run.")
        else:
            notes.append("Evaluation gates rejected this run for robustness reasons.")

        return EvalReport(
            accepted=accepted,
            primary_metric=primary_metric,
            metrics=metrics,
            grouped_metrics=grouped_metrics,
            suspicious_flags=sorted(set(suspicious_flags)),
            notes=notes,
        )

