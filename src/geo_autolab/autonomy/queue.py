from __future__ import annotations

from pathlib import Path

from geo_autolab.config_utils import dump_json, load_json

from .schemas import CandidateSpec, ExperimentRecord


class ExperimentQueue:
    def __init__(self, state_path: str | Path) -> None:
        self.state_path = Path(state_path)
        state = load_json(self.state_path, default={"pending": [], "completed": []})
        self.pending = [CandidateSpec.from_dict(item) for item in state.get("pending", [])]
        self.completed = [ExperimentRecord.from_dict(item) for item in state.get("completed", [])]

    def _persist(self) -> None:
        dump_json(
            self.state_path,
            {
                "pending": [candidate.to_dict() for candidate in self.pending],
                "completed": [record.to_dict() for record in self.completed],
            },
        )

    def enqueue_many(self, candidates: list[CandidateSpec], limit: int | None = None) -> None:
        seen = {candidate.candidate_id for candidate in self.pending}
        for candidate in sorted(candidates, key=lambda item: item.priority, reverse=True):
            if candidate.candidate_id in seen:
                continue
            self.pending.append(candidate)
            seen.add(candidate.candidate_id)
            if limit is not None and len(self.pending) >= limit:
                break
        self._persist()

    def pop_next(self) -> CandidateSpec | None:
        if not self.pending:
            return None
        self.pending.sort(key=lambda item: item.priority, reverse=True)
        candidate = self.pending.pop(0)
        self._persist()
        return candidate

    def mark_completed(self, record: ExperimentRecord) -> None:
        self.completed.append(record)
        self._persist()

    def __len__(self) -> int:
        return len(self.pending)
