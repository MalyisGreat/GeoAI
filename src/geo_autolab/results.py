from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json


@dataclass
class ExperimentResult:
    experiment_name: str
    status: str
    artifact_dir: str
    train_metrics: dict[str, float] = field(default_factory=dict)
    split_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    config_path: str = ""
    cycle_index: int = -1

    def metric(self, split: str, name: str, default: float | None = None) -> float | None:
        return self.split_metrics.get(split, {}).get(name, default)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path


@dataclass
class CycleRecord:
    cycle_index: int
    accepted: bool
    decision_score: float
    candidate_name: str
    result_path: str
    reasons: list[str] = field(default_factory=list)
    candidate_summary: dict[str, Any] = field(default_factory=dict)

    def save_jsonl(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(self)) + "\n")
        return path
