import io
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image

from geobot.data.dataset import GeoDataset, attach_label_indices
from geobot.utils.io import build_zip_member_index


def test_dataset_can_read_from_zip_index(tmp_path: Path) -> None:
    archive_root = tmp_path / "raw" / "images" / "train"
    archive_root.mkdir(parents=True, exist_ok=True)
    image_name = "123.jpg"
    archive_path = archive_root / "00.zip"

    image_buffer = io.BytesIO()
    Image.new("RGB", (32, 32), color=(10, 20, 30)).save(image_buffer, format="JPEG")
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(f"images/train/a/{image_name}", image_buffer.getvalue())

    build_zip_member_index(tmp_path / "raw", tmp_path / "raw" / "zip_member_index.parquet")
    frame = pd.DataFrame(
        [
            {
                "image_id": "123",
                "image_relpath": image_name,
                "latitude": 1.0,
                "longitude": 2.0,
                "split": "train",
                "coarse_cell": "c0",
                "fine_cell": "f0",
            }
        ]
    )
    train_frame, val_frame, _ = attach_label_indices(frame, frame)
    dataset = GeoDataset(train_frame, image_root=tmp_path / "raw", image_size=32, augment=False)
    sample = dataset[0]
    assert sample["image"].shape == (3, 32, 32)
