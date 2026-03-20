from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from geo_autolab.config import load_model
from geo_autolab.contracts import ExperimentResult

from .config import AutoLabConfig
from .loop import AutoRecycleLoop
from .runner import LocalExperimentExecutor, build_initial_spec


class AutonomousLoop:
    def __init__(self, base_config: AutoLabConfig) -> None:
        self.base_config = base_config

    def run(self) -> dict[str, Any]:
        initial_spec = build_initial_spec(self.base_config)
        executor = LocalExperimentExecutor(self.base_config)
        loop = AutoRecycleLoop(self.base_config, executor)
        result = loop.run(initial_spec)
        return {
            "best_candidate": result.best_result.spec.name,
            "accepted": result.best_result.report.accepted,
            "primary_metric": result.best_result.report.primary_metric,
            "checkpoint_path": str(result.best_result.checkpoint_path) if result.best_result.checkpoint_path else None,
            "cycles_completed": len({item.spec.cycle_index for item in result.all_results}),
            "experiments_run": len(result.all_results),
        }

    @classmethod
    def from_path(cls, path: str | Path) -> "AutonomousLoop":
        config = load_model(path, AutoLabConfig)
        return cls(base_config=config)
