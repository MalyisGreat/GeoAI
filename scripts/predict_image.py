from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from geobot.data.dataset import _pil_to_tensor
from geobot.model import GeoLocator
from geobot.utils.geo import tensor_unit_to_latlon


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a coordinate from one image.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]
    model = GeoLocator(
        backbone_name=config["model"]["backbone"],
        fallback_backbone=config["model"]["fallback_backbone"],
        pretrained=False,
        hidden_dim=int(config["model"]["hidden_dim"]),
        embedding_dim=int(config["model"]["embedding_dim"]),
        dropout=float(config["model"]["dropout"]),
        num_coarse_classes=len(checkpoint["coarse_to_idx"]),
        num_fine_classes=len(checkpoint["fine_to_idx"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    tensor = _pil_to_tensor(
        Image.open(args.image), int(config["model"]["image_size"]), augment=False
    ).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor.to(device))
        regressed = outputs["coord_unit"]
        cell_centers = checkpoint.get("cell_centers")
        if cell_centers is not None:
            fine_idx = outputs["fine_logits"].argmax(dim=-1)
            cell_prior = cell_centers.to(device)[fine_idx]
            blend = float(config["eval"]["retrieval_blend"])
            pred_unit = torch.nn.functional.normalize(
                (1.0 - blend) * regressed + blend * cell_prior, dim=-1
            )
        else:
            pred_unit = regressed
        latlon = tensor_unit_to_latlon(pred_unit).cpu().numpy()[0]

    payload = {
        "latitude": float(latlon[0]),
        "longitude": float(latlon[1]),
        "predicted_coarse_idx": int(outputs["coarse_logits"].argmax(dim=-1).item()),
        "predicted_fine_idx": int(outputs["fine_logits"].argmax(dim=-1).item()),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
