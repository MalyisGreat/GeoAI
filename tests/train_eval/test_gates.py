from __future__ import annotations

from geo_autolab.eval.config import EvalConfig
from geo_autolab.eval.gates import AntiRewardHackGate


def test_gate_accepts_balanced_metrics() -> None:
    gate = AntiRewardHackGate(EvalConfig())
    report = gate.evaluate(
        metrics={
            "median_km": 800.0,
            "within_100km": 0.2,
            "avg_confidence": 0.22,
            "geocell_top1": 0.18,
            "ece": 0.05,
        },
        grouped_metrics={
            "domain:rural": {"within_100km": 0.15},
            "domain:urban": {"within_100km": 0.22},
        },
    )
    assert report.accepted is True
    assert report.suspicious_flags == []


def test_gate_rejects_shortcut_like_pattern() -> None:
    gate = AntiRewardHackGate(EvalConfig())
    report = gate.evaluate(
        metrics={
            "median_km": 2600.0,
            "within_100km": 0.03,
            "avg_confidence": 0.65,
            "geocell_top1": 0.40,
            "ece": 0.4,
        },
        grouped_metrics={
            "domain:rural": {"within_100km": 0.01},
            "domain:urban": {"within_100km": 0.20},
        },
    )
    assert report.accepted is False
    assert "coarse-cell-shortcut-risk" in report.suspicious_flags
