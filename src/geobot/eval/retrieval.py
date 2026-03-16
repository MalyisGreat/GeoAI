from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from geobot.utils.geo import latlon_to_unit, normalize_unit_tensor


def build_cell_center_tensor(train_frame: pd.DataFrame, num_fine_classes: int) -> torch.Tensor:
    centers = np.zeros((num_fine_classes, 3), dtype=np.float32)
    grouped = train_frame.groupby("fine_idx")[["latitude", "longitude"]].mean()
    for fine_idx, row in grouped.iterrows():
        centers[int(fine_idx)] = latlon_to_unit(np.array([row["latitude"]]), np.array([row["longitude"]]))[
            0
        ].astype(np.float32)
    centers[~centers.any(axis=1)] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return torch.from_numpy(centers)


@dataclass
class GalleryIndex:
    embeddings: torch.Tensor
    coord_units: torch.Tensor
    coarse_idx: torch.Tensor

    @classmethod
    def build(
        cls,
        model: torch.nn.Module,
        dataset: torch.utils.data.Dataset,
        *,
        device: torch.device,
        batch_size: int,
        num_workers: int,
        max_items: int | None = None,
    ) -> "GalleryIndex":
        if max_items is not None and len(dataset) > max_items:
            dataset = Subset(dataset, range(max_items))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        embeddings: list[torch.Tensor] = []
        coord_units: list[torch.Tensor] = []
        coarse_idx: list[torch.Tensor] = []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device)
                outputs = model(images)
                embeddings.append(outputs["embedding"].cpu())
                coord_units.append(batch["coord_unit"].cpu())
                coarse_idx.append(batch["coarse_idx"].cpu())
        return cls(
            embeddings=torch.cat(embeddings, dim=0),
            coord_units=torch.cat(coord_units, dim=0),
            coarse_idx=torch.cat(coarse_idx, dim=0),
        )

    def query(
        self,
        query_embeddings: torch.Tensor,
        regressed_units: torch.Tensor,
        coarse_logits: torch.Tensor,
        *,
        top_k: int,
        blend: float,
        coarse_top_k: int,
    ) -> torch.Tensor:
        gallery_embeddings = self.embeddings.to(query_embeddings.device)
        gallery_units = self.coord_units.to(query_embeddings.device)
        gallery_coarse = self.coarse_idx.to(query_embeddings.device)

        similarity = query_embeddings @ gallery_embeddings.T
        if coarse_top_k > 0:
            allowed = coarse_logits.topk(k=min(coarse_top_k, coarse_logits.shape[-1]), dim=-1).indices
            coarse_mask = (gallery_coarse.unsqueeze(0) == allowed.unsqueeze(-1)).any(dim=1)
            similarity = similarity.masked_fill(~coarse_mask, float("-inf"))

        top_k = min(top_k, similarity.shape[-1])
        values, indices = similarity.topk(k=top_k, dim=-1)
        weights = torch.softmax(values / 0.05, dim=-1)
        neighbor_units = gallery_units[indices]
        retrieved = normalize_unit_tensor((neighbor_units * weights.unsqueeze(-1)).sum(dim=1))
        return normalize_unit_tensor((1.0 - blend) * regressed_units + blend * retrieved)
