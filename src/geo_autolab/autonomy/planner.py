from __future__ import annotations

import copy

from geo_autolab.contracts import ExperimentResult, ExperimentSpec

from .config import AutoLabConfig


class ExperimentPlanner:
    def __init__(self, config: AutoLabConfig) -> None:
        self.config = config

    def propose(self, anchor: ExperimentSpec, previous_best: ExperimentResult | None, cycle_index: int) -> list[ExperimentSpec]:
        base = copy.deepcopy(anchor)
        base.cycle_index = cycle_index
        base.name = f"cycle-{cycle_index:02d}-baseline"

        candidates = [base]
        candidates.append(self._make_robustness_candidate(base, cycle_index))
        candidates.append(self._make_optimizer_candidate(base, cycle_index))

        if previous_best and any("median-distance-too-high" == flag for flag in previous_best.report.suspicious_flags):
            candidates.append(self._make_capacity_candidate(base, cycle_index))

        return candidates[: self.config.candidates_per_cycle]

    def _make_robustness_candidate(self, anchor: ExperimentSpec, cycle_index: int) -> ExperimentSpec:
        candidate = copy.deepcopy(anchor)
        candidate.name = f"cycle-{cycle_index:02d}-robustness"
        aug = candidate.train["augmentation"]
        aug["random_erasing_prob"] = min(0.4, aug["random_erasing_prob"] + 0.1)
        aug["color_jitter"] = min(0.35, aug["color_jitter"] + 0.05)
        candidate.model["loss"]["label_smoothing"] = min(0.15, candidate.model["loss"]["label_smoothing"] + 0.02)
        candidate.train["learning_rate"] *= 0.85
        candidate.notes.append("Harden against shortcut learning with stronger corruption and lower step size.")
        return candidate

    def _make_optimizer_candidate(self, anchor: ExperimentSpec, cycle_index: int) -> ExperimentSpec:
        candidate = copy.deepcopy(anchor)
        candidate.name = f"cycle-{cycle_index:02d}-conservative"
        candidate.train["batch_size"] = max(6, int(candidate.train["batch_size"]) - 2)
        candidate.train["grad_accum_steps"] = int(candidate.train["grad_accum_steps"]) + 1
        candidate.train["backbone_lr_scale"] = min(0.35, float(candidate.train["backbone_lr_scale"]) + 0.05)
        candidate.notes.append("Trade throughput for stabler gradients on 8 GB VRAM.")
        return candidate

    def _make_capacity_candidate(self, anchor: ExperimentSpec, cycle_index: int) -> ExperimentSpec:
        candidate = copy.deepcopy(anchor)
        candidate.name = f"cycle-{cycle_index:02d}-capacity"
        candidate.train["learning_rate"] *= 1.1
        candidate.train["max_epochs"] = int(candidate.train["max_epochs"]) + 1
        candidate.model["adapter"]["dropout"] = max(0.05, float(candidate.model["adapter"]["dropout"]) - 0.03)
        candidate.notes.append("Lean slightly harder into capacity because distance remains poor.")
        return candidate

