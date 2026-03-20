from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from geo_autolab.autonomy.orchestrator import AutonomousLoop


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the autonomous geolocation improvement loop.")
    parser.add_argument(
        "config",
        nargs="?",
        default=str(ROOT / "configs" / "autonomy" / "local_cycle.yaml"),
        help="Path to an AutoLab YAML config",
    )
    args = parser.parse_args()

    try:
        loop = AutonomousLoop.from_path(args.config)
        summary = loop.run()
        print(json.dumps(summary, indent=2))
        return 0
    except FileNotFoundError as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
