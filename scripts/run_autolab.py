from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from geo_autolab.autonomy.config import AutoLabConfig
from geo_autolab.autonomy.loop import AutoRecycleLoop
from geo_autolab.autonomy.runner import LocalExperimentExecutor, build_initial_spec
from geo_autolab.config import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Geo AutoLab autonomous local training loop.")
    parser.add_argument(
        "--config",
        default="configs/autonomy/local_cycle.yaml",
        help="Path to the autonomy config YAML.",
    )
    args = parser.parse_args()

    try:
        autolab_config = load_model(args.config, AutoLabConfig)
        initial_spec = build_initial_spec(autolab_config)
        executor = LocalExperimentExecutor(autolab_config)
        loop = AutoRecycleLoop(autolab_config, executor)
        result = loop.run(initial_spec)
        print(
            f"best={result.best_result.spec.name} "
            f"accepted={result.best_result.report.accepted} "
            f"median_km={result.best_result.report.primary_metric:.2f}"
        )
    except FileNotFoundError as exc:
        print(f"error={exc}")


if __name__ == "__main__":
    main()
