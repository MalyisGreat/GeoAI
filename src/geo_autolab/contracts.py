from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ExperimentSpec:
    name: str
    cycle_index: int
    model: dict[str, Any]
    train: dict[str, Any]
    evaluation: dict[str, Any]
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EvalReport:
    accepted: bool
    primary_metric: float
    metrics: dict[str, float]
    grouped_metrics: dict[str, dict[str, float]]
    suspicious_flags: list[str]
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ExperimentResult:
    spec: ExperimentSpec
    report: EvalReport
    checkpoint_path: Path | None = None

