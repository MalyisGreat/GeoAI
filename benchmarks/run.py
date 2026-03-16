from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from geobot.config import load_config
from geobot.model import GeoLocator


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small forward/backward benchmark.")
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--output", default="benchmarks/latest-metrics.json")
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(config["runtime"].get("device", "auto"))
    batch_size = int(config["train"]["batch_size"])
    image_size = int(config["model"]["image_size"])
    model = GeoLocator(
        backbone_name=config["model"]["backbone"],
        fallback_backbone=config["model"]["fallback_backbone"],
        pretrained=bool(config["model"].get("pretrained", False)),
        hidden_dim=int(config["model"]["hidden_dim"]),
        embedding_dim=int(config["model"]["embedding_dim"]),
        dropout=float(config["model"]["dropout"]),
        num_coarse_classes=32,
        num_fine_classes=128,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sample = torch.randn(batch_size, 3, image_size, image_size, device=device)
    coarse_target = torch.randint(0, 32, (batch_size,), device=device)
    fine_target = torch.randint(0, 128, (batch_size,), device=device)
    coord_target = torch.nn.functional.normalize(torch.randn(batch_size, 3, device=device), dim=-1)

    torch.set_grad_enabled(True)
    start = time.perf_counter()
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        out = model(sample)
        loss = (
            torch.nn.functional.cross_entropy(out["coarse_logits"], coarse_target)
            + torch.nn.functional.cross_entropy(out["fine_logits"], fine_target)
            + (1.0 - (out["coord_unit"] * coord_target).sum(dim=-1)).mean()
        )
        loss.backward()
        optimizer.step()
    elapsed = time.perf_counter() - start
    metrics = {
        "device": str(device),
        "steps": args.steps,
        "batch_size": batch_size,
        "image_size": image_size,
        "avg_step_ms": (elapsed / args.steps) * 1000.0,
        "samples_per_sec": (batch_size * args.steps) / max(elapsed, 1e-6),
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
