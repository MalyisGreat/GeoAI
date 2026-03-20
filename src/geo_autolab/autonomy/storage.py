from __future__ import annotations

from pathlib import Path

from geo_autolab.contracts import ExperimentResult
from geo_autolab.utils import append_jsonl, ensure_dir


class RunRegistry:
    def __init__(self, history_path: str) -> None:
        self.history_path = Path(history_path)
        ensure_dir(self.history_path.parent)

    def record(self, result: ExperimentResult) -> None:
        append_jsonl(
            self.history_path,
            {
                "name": result.spec.name,
                "cycle_index": result.spec.cycle_index,
                "primary_metric": result.report.primary_metric,
                "accepted": result.report.accepted,
                "flags": result.report.suspicious_flags,
                "checkpoint_path": str(result.checkpoint_path) if result.checkpoint_path else None,
                "metrics": result.report.metrics,
            },
        )

