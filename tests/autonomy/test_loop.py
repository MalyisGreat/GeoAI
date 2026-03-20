from __future__ import annotations

from geo_autolab.autonomy.config import AutoLabConfig
from geo_autolab.autonomy.loop import AutoRecycleLoop
from geo_autolab.contracts import EvalReport, ExperimentResult, ExperimentSpec


class FakeExecutor:
    def run(self, spec: ExperimentSpec) -> ExperimentResult:
        if "robustness" in spec.name:
            primary = 250.0
            accepted = True
        elif "conservative" in spec.name:
            primary = 400.0
            accepted = True
        else:
            primary = 600.0
            accepted = False
        report = EvalReport(
            accepted=accepted,
            primary_metric=primary,
            metrics={"median_km": primary, "within_100km": 0.1},
            grouped_metrics={},
            suspicious_flags=[] if accepted else ["median-distance-too-high"],
            notes=[],
        )
        return ExperimentResult(spec=spec, report=report, checkpoint_path=None)


def test_auto_loop_selects_best_accepted_candidate(tmp_path) -> None:
    config = AutoLabConfig(
        run_root=str(tmp_path / "runs"),
        history_path=str(tmp_path / "history.jsonl"),
        max_cycles=2,
        candidates_per_cycle=3,
    )
    loop = AutoRecycleLoop(config, executor=FakeExecutor())
    initial = ExperimentSpec(
        name="seed",
        cycle_index=0,
        model={"loss": {"label_smoothing": 0.05}, "adapter": {"dropout": 0.1}},
        train={
            "batch_size": 12,
            "grad_accum_steps": 2,
            "learning_rate": 0.0003,
            "backbone_lr_scale": 0.2,
            "max_epochs": 5,
            "augmentation": {
                "resize_scale_min": 0.7,
                "color_jitter": 0.2,
                "blur_prob": 0.15,
                "grayscale_prob": 0.05,
                "random_erasing_prob": 0.15,
            },
        },
        evaluation={},
        notes=[],
    )

    result = loop.run(initial)

    assert result.best_result.report.accepted is True
    assert "robustness" in result.best_result.spec.name

