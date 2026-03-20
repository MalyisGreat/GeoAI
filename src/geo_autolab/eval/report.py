from __future__ import annotations

from geo_autolab.contracts import EvalReport


def summarize_report(report: EvalReport) -> str:
    lines = [
        f"accepted={report.accepted}",
        f"median_km={report.metrics.get('median_km', 0.0):.2f}",
        f"within_100km={report.metrics.get('within_100km', 0.0):.3f}",
    ]
    if report.suspicious_flags:
        lines.append("flags=" + ",".join(report.suspicious_flags))
    return " ".join(lines)
