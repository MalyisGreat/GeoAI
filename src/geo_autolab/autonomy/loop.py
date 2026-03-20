from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from geo_autolab.contracts import ExperimentResult, ExperimentSpec

from .config import AutoLabConfig
from .planner import ExperimentPlanner
from .storage import RunRegistry


class ExperimentExecutor(Protocol):
    def run(self, spec: ExperimentSpec) -> ExperimentResult:
        ...


@dataclass(slots=True)
class AutoLoopResult:
    best_result: ExperimentResult
    all_results: list[ExperimentResult] = field(default_factory=list)


class AutoRecycleLoop:
    def __init__(self, config: AutoLabConfig, executor: ExperimentExecutor) -> None:
        self.config = config
        self.executor = executor
        self.planner = ExperimentPlanner(config)
        self.registry = RunRegistry(config.history_path)

    def _target_reached(self, result: ExperimentResult) -> bool:
        metrics = result.report.metrics
        if self.config.target_median_km is not None:
            if result.report.primary_metric > self.config.target_median_km:
                return False
        if self.config.target_within_100km is not None:
            if metrics.get("within_100km", 0.0) < self.config.target_within_100km:
                return False
        if self.config.target_geocell_top1 is not None:
            if metrics.get("geocell_top1", 0.0) < self.config.target_geocell_top1:
                return False
        return any(
            target is not None
            for target in (
                self.config.target_median_km,
                self.config.target_within_100km,
                self.config.target_geocell_top1,
            )
        )

    def run(self, initial_spec: ExperimentSpec) -> AutoLoopResult:
        anchor = initial_spec
        previous_best: ExperimentResult | None = None
        global_best: ExperimentResult | None = None
        all_results: list[ExperimentResult] = []
        cycle_index = 0
        while True:
            if not self.config.continue_until_target and cycle_index >= self.config.max_cycles:
                break
            candidates = self.planner.propose(anchor, previous_best, cycle_index=cycle_index)
            cycle_results = [self.executor.run(candidate) for candidate in candidates]
            for result in cycle_results:
                self.registry.record(result)
                all_results.append(result)

            accepted = [result for result in cycle_results if result.report.accepted]
            cycle_best = min(accepted or cycle_results, key=lambda item: item.report.primary_metric)
            if global_best is None or cycle_best.report.primary_metric < global_best.report.primary_metric:
                global_best = cycle_best
            previous_best = global_best
            anchor = global_best.spec
            cycle_index += 1

            if self._target_reached(global_best):
                break

        if global_best is None:
            raise RuntimeError("Autonomy loop produced no results.")
        return AutoLoopResult(best_result=global_best, all_results=all_results)
