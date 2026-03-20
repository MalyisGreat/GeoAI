from __future__ import annotations

import argparse

from .engine import AutonomyEngine
from .schemas import AutonomyConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the autonomous geolocation optimizer")
    parser.add_argument("--config", required=True, help="Path to autonomy YAML")
    parser.add_argument("--cycles", type=int, default=0, help="Override max cycle count")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AutonomyConfig.from_yaml(args.config)
    engine = AutonomyEngine(config)
    results = engine.run(cycles=args.cycles or None)
    print({"cycles_run": len(results), "best": results[-1].candidate.candidate_id if results else None})


if __name__ == "__main__":
    main()
