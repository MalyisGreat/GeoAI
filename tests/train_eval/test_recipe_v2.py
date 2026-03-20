from __future__ import annotations

import csv
import json

import torch
from PIL import Image

from geo_autolab.geo import latlon_to_unit_xyz
from geo_autolab.models.config import LossConfig
from geo_autolab.models.heads import MultiTaskGeoHead
from geo_autolab.models.losses import GeoCriterion
from geo_autolab.train.config import TrainConfig
from geo_autolab.train.dataset import GeoDataset, load_manifest
from geo_autolab.train.geocells import compute_hierarchy_info


def test_geocell_conditioned_head_decodes_to_selected_centroid() -> None:
    head = MultiTaskGeoHead(
        input_dim=4,
        hidden_dim=4,
        embedding_dim=2,
        country_classes=2,
        region_classes=2,
        geocell_classes=2,
        dropout=0.0,
        predict_country=True,
        predict_region=True,
        predict_geocell=True,
        predict_uncertainty=False,
        decode_topk=1,
        max_offset_norm=0.08,
        decode_confidence_threshold=0.08,
        decode_confidence_sharpness=18.0,
    )
    head.set_geocell_centroids(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32))
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()
        assert head.country_head is not None
        assert head.region_head is not None
        assert head.geocell_head is not None
        head.country_head.bias[1] = 8.0
        head.region_head.bias[1] = 8.0
        head.geocell_head.bias[1] = 10.0

    outputs = head(torch.zeros((1, 4), dtype=torch.float32))
    assert outputs["country_logits"] is not None
    assert outputs["region_logits"] is not None
    assert outputs["unit_xyz"] is not None
    assert torch.allclose(outputs["unit_xyz"], torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-4)


def test_spatial_geocell_loss_prefers_neighboring_cells_over_far_cells() -> None:
    criterion = GeoCriterion(
        LossConfig(
            country_weight=0.0,
            region_weight=0.0,
            geocell_weight=1.0,
            geodesic_weight=0.0,
            embedding_weight=0.0,
            offset_weight=0.0,
            hierarchy_consistency_weight=0.0,
            spatial_geocell_radius_km=250.0,
            spatial_geocell_topk=2,
        )
    )
    centroids = torch.stack(
        [
            latlon_to_unit_xyz(torch.tensor([0.0, 0.0], dtype=torch.float32)),
            latlon_to_unit_xyz(torch.tensor([0.0, 1.0], dtype=torch.float32)),
            latlon_to_unit_xyz(torch.tensor([0.0, 180.0], dtype=torch.float32)),
        ]
    )
    criterion.set_geocell_centroids(centroids)

    batch = {
        "unit_xyz": centroids[[0]],
        "geocell_id": torch.tensor([0], dtype=torch.long),
        "country_id": torch.tensor([0], dtype=torch.long),
        "region_id": torch.tensor([0], dtype=torch.long),
    }
    shared = {
        "embedding": None,
        "uncertainty": None,
        "country_logits": None,
        "region_logits": None,
        "country_probs": None,
        "region_probs": None,
        "base_centroid": centroids[[0]],
        "local_offset": torch.zeros((1, 3), dtype=torch.float32),
        "geocell_probs": torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
        "unit_xyz": centroids[[0]],
    }

    near_loss = criterion(
        {
            **shared,
            "geocell_logits": torch.tensor([[2.0, 4.0, -8.0]], dtype=torch.float32),
        },
        batch,
    ).geocell
    far_loss = criterion(
        {
            **shared,
            "geocell_logits": torch.tensor([[-8.0, -8.0, 4.0]], dtype=torch.float32),
        },
        batch,
    ).geocell

    assert near_loss.item() < far_loss.item()


def test_hierarchy_info_and_dataset_emit_country_region_ids(tmp_path) -> None:
    image_a = tmp_path / "a.jpg"
    image_b = tmp_path / "b.jpg"
    Image.new("RGB", (64, 64), (255, 0, 0)).save(image_a)
    Image.new("RGB", (64, 64), (0, 255, 0)).save(image_b)

    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", "latitude", "longitude", "geocell_id", "split", "domain", "source"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "image_path": str(image_a),
                "latitude": 35.0,
                "longitude": -97.0,
                "geocell_id": 1,
                "split": "train",
                "domain": "street",
                "source": "osv5m",
            }
        )
        writer.writerow(
            {
                "image_path": str(image_b),
                "latitude": 48.8,
                "longitude": 2.3,
                "geocell_id": 2,
                "split": "train",
                "domain": "street",
                "source": "osv5m",
            }
        )

    metadata = tmp_path / "metadata.jsonl"
    with metadata.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"image_path": str(image_a), "country": "US", "region": "Oklahoma", "sub_region": ""}) + "\n")
        handle.write(json.dumps({"image_path": str(image_b), "country": "FR", "region": "Ile-de-France", "sub_region": ""}) + "\n")

    hierarchy_info = compute_hierarchy_info(manifest, geocell_classes=3)
    config = TrainConfig(batch_size=2, eval_batch_size=2, num_workers=0)
    records = load_manifest(manifest, split="train", train_config=config, hierarchy_info=hierarchy_info)
    dataset = GeoDataset(records, image_size=32, train_config=config, training=True)
    sample = dataset[0]

    assert len(hierarchy_info.country_labels) >= 3
    assert len(hierarchy_info.region_labels) >= 3
    assert sample["country_id"].item() in hierarchy_info.country_to_id.values()
    assert sample["region_id"].item() in hierarchy_info.region_to_id.values()
    assert "positive_country_id" in sample
    assert str(sample["group_label"]).startswith("geo_region:")


def test_hierarchy_consistency_loss_is_positive_when_heads_disagree() -> None:
    criterion = GeoCriterion(
        LossConfig(
            country_weight=0.0,
            region_weight=0.0,
            geocell_weight=0.0,
            geodesic_weight=0.0,
            embedding_weight=0.0,
            offset_weight=0.0,
            hierarchy_consistency_weight=1.0,
        )
    )
    criterion.set_hierarchy(
        geocell_to_country=torch.tensor([0, 1], dtype=torch.long),
        geocell_to_region=torch.tensor([0, 1], dtype=torch.long),
        country_classes=2,
        region_classes=2,
    )
    outputs = {
        "embedding": None,
        "unit_xyz": torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
        "fallback_unit_xyz": torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
        "country_logits": torch.tensor([[4.0, -4.0]], dtype=torch.float32),
        "country_probs": None,
        "region_logits": torch.tensor([[4.0, -4.0]], dtype=torch.float32),
        "region_probs": None,
        "geocell_logits": torch.tensor([[-4.0, 4.0]], dtype=torch.float32),
        "geocell_probs": torch.tensor([[0.0, 1.0]], dtype=torch.float32),
        "base_centroid": torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
        "local_offset": torch.zeros((1, 3), dtype=torch.float32),
        "decode_mix": torch.tensor([[1.0]], dtype=torch.float32),
        "uncertainty": None,
    }
    batch = {
        "unit_xyz": torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
        "geocell_id": torch.tensor([1], dtype=torch.long),
        "country_id": torch.tensor([1], dtype=torch.long),
        "region_id": torch.tensor([1], dtype=torch.long),
    }
    losses = criterion(outputs, batch)
    assert losses.hierarchy.item() > 0.0
