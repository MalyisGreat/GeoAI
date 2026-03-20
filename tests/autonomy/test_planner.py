from __future__ import annotations

from geo_autolab.autonomy.config import AutoLabConfig
from geo_autolab.autonomy.planner import ExperimentPlanner
from geo_autolab.contracts import ExperimentSpec
from geo_autolab.eval.config import EvalConfig
from geo_autolab.train.config import TrainConfig


def test_planner_emits_distinct_candidates() -> None:
    config = AutoLabConfig(train=TrainConfig.local_default(), evaluation=EvalConfig())
    planner = ExperimentPlanner(config)
    anchor = ExperimentSpec(
        name="seed",
        cycle_index=0,
        model={
            "loss": {"label_smoothing": 0.05},
            "adapter": {"dropout": 0.1},
        },
        train=TrainConfig.local_default().model_dump(),
        evaluation=EvalConfig().model_dump(),
        notes=[],
    )

    candidates = planner.propose(anchor, previous_best=None, cycle_index=1)

    assert len(candidates) == 3
    assert candidates[1].train["augmentation"]["random_erasing_prob"] > anchor.train["augmentation"]["random_erasing_prob"]
    assert candidates[2].train["grad_accum_steps"] > anchor.train["grad_accum_steps"]

