from pathlib import Path

import pandas as pd
from PIL import Image

from geobot.data.dataset import GeoDataset, attach_label_indices, split_train_val
from geobot.model import GeoLocator


def test_dataset_and_model_forward(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 0)]):
        file_name = f"{index}.jpg"
        Image.new("RGB", (64, 64), color=color).save(image_root / file_name)
        rows.append(
            {
                "image_id": str(index),
                "image_relpath": file_name,
                "latitude": 10.0 + index,
                "longitude": 20.0 + index,
                "split": "train",
                "coarse_cell": f"c{index % 2}",
                "fine_cell": f"f{index}",
            }
        )

    frame = pd.DataFrame(rows)
    train_frame, val_frame = split_train_val(frame, val_fraction=0.25, seed=7)
    train_frame, val_frame, _ = attach_label_indices(train_frame, val_frame)
    dataset = GeoDataset(train_frame, image_root=image_root, image_size=64, augment=False)
    batch = dataset[0]
    assert batch["image"].shape == (3, 64, 64)

    model = GeoLocator(
        backbone_name="conv_geo_tiny",
        fallback_backbone="conv_geo_tiny",
        pretrained=False,
        hidden_dim=64,
        embedding_dim=32,
        dropout=0.0,
        num_coarse_classes=2,
        num_fine_classes=len(train_frame["fine_idx"].unique()),
    )
    output = model(batch["image"].unsqueeze(0))
    assert output["embedding"].shape == (1, 32)
    assert output["coarse_logits"].shape == (1, 2)
