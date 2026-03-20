from __future__ import annotations

from pathlib import Path

from geo_autolab.autonomy.config import AutoLabConfig
from geo_autolab.autonomy.loop import AutoRecycleLoop
from geo_autolab.contracts import EvalReport, ExperimentResult, ExperimentSpec


class FakeExecutor:
    def __init__(self, root: Path) -> None:
        self.root = root

    def run(self, spec: ExperimentSpec) -> ExperimentResult:
        accepted = "robustness" in spec.name or "baseline" in spec.name
        primary_metric = 850.0 if "robustness" in spec.name else 1000.0
        report = EvalReport(
            accepted=accepted,
            primary_metric=primary_metric,
            metrics={
                "median_km": primary_metric,
                "within_100km": 0.18 if accepted else 0.05,
                "geocell_top1": 0.16 if accepted else 0.06,
                "avg_confidence": 0.18 if accepted else 0.4,
                "ece": 0.08 if accepted else 0.25,
            },
            grouped_metrics={"domain:test": {"within_100km": 0.15 if accepted else 0.02}},
            suspicious_flags=[] if accepted else ["weak-local-accuracy"],
        )
        checkpoint = self.root / f"{spec.name}.pt"
        checkpoint.write_text("fake", encoding="utf-8")
        return ExperimentResult(spec=spec, report=report, checkpoint_path=checkpoint)


def test_autorecycle_loop_records_history(tmp_path: Path) -> None:
    config = AutoLabConfig(
        history_path=str(tmp_path / "history.jsonl"),
        max_cycles=2,
        candidates_per_cycle=2,
    )
    initial_spec = ExperimentSpec(
        name="bootstrap",
        cycle_index=0,
        model={
            "loss": {"label_smoothing": 0.05},
            "adapter": {"dropout": 0.1},
        },
        train={
            "augmentation": {"random_erasing_prob": 0.1, "color_jitter": 0.1},
            "learning_rate": 3e-4,
            "batch_size": 8,
            "grad_accum_steps": 2,
            "backbone_lr_scale": 0.2,
            "max_epochs": 2,
        },
        evaluation={},
    )
    loop = AutoRecycleLoop(config, FakeExecutor(tmp_path))
    result = loop.run(initial_spec)
    history_path = Path(config.history_path)

    assert history_path.exists()
    assert len(result.all_results) == 4
    assert result.best_result.report.primary_metric == 850.0


def test_autorecycle_loop_keeps_global_best_across_regressing_cycles(tmp_path: Path) -> None:
    config = AutoLabConfig(
        history_path=str(tmp_path / "history.jsonl"),
        max_cycles=2,
        candidates_per_cycle=1,
    )

    class RegressingExecutor:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, spec: ExperimentSpec) -> ExperimentResult:
            metrics = [700.0, 1200.0]
            metric = metrics[self.calls]
            self.calls += 1
            report = EvalReport(
                accepted=False,
                primary_metric=metric,
                metrics={"median_km": metric, "within_100km": 0.0, "geocell_top1": 0.0, "avg_confidence": 0.0, "ece": 0.0},
                grouped_metrics={"domain:test": {"within_100km": 0.0}},
                suspicious_flags=["weak-local-accuracy"],
            )
            return ExperimentResult(spec=spec, report=report, checkpoint_path=None)

    initial_spec = ExperimentSpec(
        name="bootstrap",
        cycle_index=0,
        model={"loss": {"label_smoothing": 0.05}, "adapter": {"dropout": 0.1}},
        train={
            "augmentation": {"random_erasing_prob": 0.1, "color_jitter": 0.1},
            "learning_rate": 3e-4,
            "batch_size": 8,
            "grad_accum_steps": 2,
            "backbone_lr_scale": 0.2,
            "max_epochs": 2,
        },
        evaluation={},
    )
    result = AutoRecycleLoop(config, RegressingExecutor()).run(initial_spec)
    assert result.best_result.report.primary_metric == 700.0
