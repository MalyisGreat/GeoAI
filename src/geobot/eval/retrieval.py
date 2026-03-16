from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from geobot.data.dataset import LabelMaps
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
class CellStateStore:
    centers: torch.Tensor
    clue_features: torch.Tensor


def build_cell_state_store(
    train_frame: pd.DataFrame,
    label_maps: LabelMaps,
    num_fine_classes: int,
) -> CellStateStore:
    centers = build_cell_center_tensor(train_frame, num_fine_classes)
    feature_parts: list[np.ndarray] = []

    for safe_name, mapping in label_maps.aux_categorical.items():
        column = f"aux_{safe_name}_idx"
        if column not in train_frame.columns:
            continue
        matrix = np.zeros((num_fine_classes, len(mapping)), dtype=np.float32)
        grouped = train_frame.groupby(["fine_idx", column]).size().reset_index(name="count")
        for fine_idx, aux_idx, count in grouped[["fine_idx", column, "count"]].to_numpy():
            matrix[int(fine_idx), int(aux_idx)] = float(count)
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = matrix / np.clip(row_sums, 1.0, None)
        feature_parts.append(matrix)

    for safe_name in label_maps.aux_numeric_stats:
        column = f"aux_{safe_name}_value"
        if column not in train_frame.columns:
            continue
        matrix = np.zeros((num_fine_classes, 1), dtype=np.float32)
        grouped = train_frame.groupby("fine_idx")[column].mean()
        matrix[grouped.index.to_numpy(dtype=int), 0] = grouped.to_numpy(dtype=np.float32)
        feature_parts.append(matrix)

    if feature_parts:
        clue_features = np.concatenate(feature_parts, axis=1)
    else:
        clue_features = np.zeros((num_fine_classes, 0), dtype=np.float32)
    return CellStateStore(centers=centers, clue_features=torch.from_numpy(clue_features))


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
                embeddings.append(outputs.get("retrieval_embedding", outputs["embedding"]).cpu())
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


def rerank_candidate_cells(
    outputs: dict[str, torch.Tensor],
    cell_store: CellStateStore,
    *,
    candidate_top_k: int,
    retrieval_weight: float,
    clue_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    fine_logits = outputs["fine_logits"]
    retrieval_logits = outputs.get("retrieval_logits")
    combined = fine_logits if retrieval_logits is None else fine_logits + retrieval_weight * retrieval_logits
    candidate_top_k = min(candidate_top_k, combined.shape[-1])
    scores, indices = combined.topk(candidate_top_k, dim=-1)

    clue_vector = outputs.get("clue_vector")
    clue_features = cell_store.clue_features.to(scores.device)
    if clue_vector is not None and clue_vector.shape[-1] > 0 and clue_features.shape[-1] == clue_vector.shape[-1]:
        normalized_query = F.normalize(clue_vector, dim=-1)
        normalized_features = F.normalize(clue_features, dim=-1)
        candidate_features = normalized_features[indices]
        clue_scores = (normalized_query.unsqueeze(1) * candidate_features).sum(dim=-1)
        scores = scores + clue_weight * clue_scores

    best_positions = scores.argmax(dim=-1)
    best_indices = indices[torch.arange(indices.shape[0], device=indices.device), best_positions]
    candidate_units = cell_store.centers.to(scores.device)[best_indices]
    return candidate_units, scores
