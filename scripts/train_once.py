from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from geo_autolab.autonomy.config import AutoLabConfig
from geo_autolab.autonomy.runner import LocalExperimentExecutor, build_initial_spec
from geo_autolab.config import load_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a single bootstrap Geo AutoLab experiment.")
    parser.add_argument(
        "config",
        nargs="?",
        default=str(ROOT / "configs" / "autonomy" / "local_cycle.yaml"),
        help="Path to an AutoLab YAML config",
    )
    args = parser.parse_args()

    try:
        config = load_model(args.config, AutoLabConfig)
        executor = LocalExperimentExecutor(config)
        result = executor.run(build_initial_spec(config))
        print(
            json.dumps(
                {
                    "name": result.spec.name,
                    "accepted": result.report.accepted,
                    "primary_metric": result.report.primary_metric,
                    "checkpoint_path": str(result.checkpoint_path) if result.checkpoint_path else None,
                },
                indent=2,
            )
        )
        return 0
    except FileNotFoundError as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
