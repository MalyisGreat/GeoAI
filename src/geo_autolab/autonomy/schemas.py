from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from geo_autolab.config_utils import dump_yaml, load_yaml
from geo_autolab.paths import ROOT


class GuardConfig(BaseModel):
    min_geocell_gain: float = 0.01
    min_geodesic_improvement_km: float = 5.0
    max_country_regression: float = 0.02
    max_calibration_ece: float = 0.18
    max_stress_drop: float = 0.18
    max_shortcut_risk: float = 0.35


class RunnerConfig(BaseModel):
    python_executable: str = "python"
    dry_run: bool = False


class AutonomyConfig(BaseModel):
    base_experiment_config: str = str(ROOT / "configs" / "model" / "base.yaml")
    output_root: str = str(ROOT / "runs" / "autonomy")
    max_cycles: int = 6
    fanout_per_cycle: int = 4
    max_queue_size: int = 8
    recycle_budget: int = 1
    guard: GuardConfig = Field(default_factory=GuardConfig)
    runner: RunnerConfig = Field(default_factory=RunnerConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AutonomyConfig":
        return cls.model_validate(load_yaml(path))

    def to_yaml(self, path: str | Path) -> None:
        dump_yaml(path, self.model_dump(mode="json"))


@dataclass
class CandidateSpec:
    candidate_id: str
    manager: str
    subagent: str
    rationale: str
    overrides: dict[str, Any]
    priority: float = 0.0
    generation: int = 0
    retries: int = 0
    parent_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CandidateSpec":
        return cls(**data)


@dataclass
class PromotionDecision:
    promote: bool
    score: float
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentRecord:
    candidate: CandidateSpec
    train_summary: dict[str, Any]
    eval_summary: dict[str, Any]
    analysis: dict[str, Any]
    decision: PromotionDecision
    run_dir: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "train_summary": self.train_summary,
            "eval_summary": self.eval_summary,
            "analysis": self.analysis,
            "decision": self.decision.to_dict(),
            "run_dir": self.run_dir,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentRecord":
        return cls(
            candidate=CandidateSpec.from_dict(data["candidate"]),
            train_summary=data["train_summary"],
            eval_summary=data["eval_summary"],
            analysis=data["analysis"],
            decision=PromotionDecision(**data["decision"]),
            run_dir=data["run_dir"],
        )


@dataclass
class MutationProposal:
    manager: str
    subagent: str
    rationale: str
    overrides: dict[str, Any]
    priority: float
    parent_id: str | None = None
    generation: int = 0

    def to_candidate(self, sequence: int) -> CandidateSpec:
        candidate_id = f"{self.manager}-{self.subagent}-{sequence:04d}"
        return CandidateSpec(
            candidate_id=candidate_id,
            manager=self.manager,
            subagent=self.subagent,
            rationale=self.rationale,
            overrides=self.overrides,
            priority=self.priority,
            generation=self.generation,
            parent_id=self.parent_id,
        )
