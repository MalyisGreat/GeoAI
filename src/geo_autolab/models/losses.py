from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from geo_autolab.geo import great_circle_distance_km

from .config import LossConfig


@dataclass(slots=True)
class LossBreakdown:
    total: Tensor
    country: Tensor
    region: Tensor
    geocell: Tensor
    geodesic: Tensor
    embedding: Tensor
    offset: Tensor
    hierarchy: Tensor


class GeoCriterion(nn.Module):
    def __init__(self, config: LossConfig) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("geocell_centroids", torch.empty((0, 3), dtype=torch.float32), persistent=False)
        self.register_buffer("geocell_country_membership", torch.empty((0, 0), dtype=torch.float32), persistent=False)
        self.register_buffer("geocell_region_membership", torch.empty((0, 0), dtype=torch.float32), persistent=False)

    def set_geocell_centroids(self, centroids: Tensor) -> None:
        if centroids.ndim != 2 or centroids.shape[-1] != 3:
            raise ValueError(f"Expected centroids with shape [classes, 3], got {tuple(centroids.shape)}")
        self.geocell_centroids = nn.functional.normalize(centroids.detach().clone(), dim=-1)

    def set_hierarchy(self, geocell_to_country: Tensor, geocell_to_region: Tensor, country_classes: int, region_classes: int) -> None:
        geocell_classes = geocell_to_country.shape[0]
        country_membership = torch.zeros((geocell_classes, country_classes), dtype=torch.float32)
        region_membership = torch.zeros((geocell_classes, region_classes), dtype=torch.float32)
        country_membership.scatter_(1, geocell_to_country.view(-1, 1), 1.0)
        region_membership.scatter_(1, geocell_to_region.view(-1, 1), 1.0)
        self.geocell_country_membership = country_membership
        self.geocell_region_membership = region_membership

    def _spatial_geocell_loss(self, logits: Tensor, target_xyz: Tensor, target_ids: Tensor) -> Tensor:
        if self.geocell_centroids.numel() == 0 or self.config.spatial_geocell_radius_km <= 0.0:
            return nn.functional.cross_entropy(
                logits,
                target_ids,
                label_smoothing=self.config.label_smoothing,
            )

        cosine = target_xyz @ self.geocell_centroids.T
        distances = torch.arccos(cosine.clamp(-1.0 + 1e-6, 1.0 - 1e-6)) * 6371.0088

        if 0 < self.config.spatial_geocell_topk < distances.shape[-1]:
            nearest_distances, nearest_indices = distances.topk(self.config.spatial_geocell_topk, dim=-1, largest=False)
            weights = torch.zeros_like(distances)
            nearest_weights = torch.exp(-((nearest_distances / self.config.spatial_geocell_radius_km) ** 2))
            weights.scatter_(1, nearest_indices, nearest_weights)
        else:
            weights = torch.exp(-((distances / self.config.spatial_geocell_radius_km) ** 2))

        weights.scatter_add_(
            1,
            target_ids.unsqueeze(1),
            torch.ones((target_ids.shape[0], 1), dtype=weights.dtype, device=weights.device),
        )
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        return -(weights * log_probs).sum(dim=-1).mean()

    def _contrastive_embedding_loss(self, anchor_embedding: Tensor, positive_embedding: Tensor) -> Tensor:
        logits = anchor_embedding @ positive_embedding.T
        logits = logits / self.config.contrastive_temperature
        labels = torch.arange(anchor_embedding.shape[0], device=anchor_embedding.device)
        return 0.5 * (
            nn.functional.cross_entropy(logits, labels) + nn.functional.cross_entropy(logits.T, labels)
        )

    def _offset_loss(self, outputs: dict[str, Tensor | None], target_xyz: Tensor) -> Tensor:
        if outputs["local_offset"] is None or outputs["base_centroid"] is None:
            return torch.zeros((), device=target_xyz.device)
        base_centroid = outputs["base_centroid"]
        target_offset = target_xyz - torch.sum(target_xyz * base_centroid, dim=-1, keepdim=True) * base_centroid
        return nn.functional.smooth_l1_loss(outputs["local_offset"], target_offset)

    def _hierarchy_consistency_loss(self, outputs: dict[str, Tensor | None]) -> Tensor:
        geocell_probs = outputs.get("geocell_probs")
        if geocell_probs is None:
            return torch.zeros((), device=self.geocell_centroids.device if self.geocell_centroids.numel() else torch.device("cpu"))

        loss = torch.zeros((), device=geocell_probs.device)
        if outputs.get("country_logits") is not None and self.geocell_country_membership.numel() > 0:
            target_country = (geocell_probs @ self.geocell_country_membership.to(geocell_probs.device)).clamp_min(1e-6)
            target_country = target_country / target_country.sum(dim=-1, keepdim=True)
            loss = loss + nn.functional.kl_div(
                nn.functional.log_softmax(outputs["country_logits"], dim=-1),
                target_country,
                reduction="batchmean",
            )
        if outputs.get("region_logits") is not None and self.geocell_region_membership.numel() > 0:
            target_region = (geocell_probs @ self.geocell_region_membership.to(geocell_probs.device)).clamp_min(1e-6)
            target_region = target_region / target_region.sum(dim=-1, keepdim=True)
            loss = loss + nn.functional.kl_div(
                nn.functional.log_softmax(outputs["region_logits"], dim=-1),
                target_region,
                reduction="batchmean",
            )
        return loss

    def forward(
        self,
        outputs: dict[str, Tensor | None],
        batch: dict[str, Tensor],
        positive_outputs: dict[str, Tensor | None] | None = None,
    ) -> LossBreakdown:
        device = batch["unit_xyz"].device
        country_loss = torch.zeros((), device=device)
        if outputs["country_logits"] is not None and "country_id" in batch and outputs["country_logits"].shape[-1] > 1:
            country_loss = nn.functional.cross_entropy(
                outputs["country_logits"],
                batch["country_id"],
                label_smoothing=self.config.label_smoothing,
            )

        region_loss = torch.zeros((), device=device)
        if outputs["region_logits"] is not None and "region_id" in batch and outputs["region_logits"].shape[-1] > 1:
            region_loss = nn.functional.cross_entropy(
                outputs["region_logits"],
                batch["region_id"],
                label_smoothing=self.config.label_smoothing,
            )

        geocell_loss = torch.zeros((), device=device)
        if outputs["geocell_logits"] is not None and "geocell_id" in batch:
            geocell_loss = self._spatial_geocell_loss(
                outputs["geocell_logits"],
                batch["unit_xyz"],
                batch["geocell_id"],
            )

        geodesic_km = great_circle_distance_km(outputs["unit_xyz"], batch["unit_xyz"])
        if outputs["uncertainty"] is not None:
            scale = outputs["uncertainty"].squeeze(-1)
            geodesic_loss = (geodesic_km / scale + scale.log()).mean()
        else:
            geodesic_loss = geodesic_km.mean()

        embedding_loss = torch.zeros((), device=device)
        if (
            positive_outputs is not None
            and outputs["embedding"] is not None
            and positive_outputs["embedding"] is not None
            and outputs["embedding"].shape[0] > 0
        ):
            embedding_loss = self._contrastive_embedding_loss(outputs["embedding"], positive_outputs["embedding"])

        offset_loss = self._offset_loss(outputs, batch["unit_xyz"])
        hierarchy_loss = self._hierarchy_consistency_loss(outputs).to(device)

        total = (
            self.config.country_weight * country_loss
            + self.config.region_weight * region_loss
            + self.config.geocell_weight * geocell_loss
            + self.config.geodesic_weight * geodesic_loss
            + self.config.embedding_weight * embedding_loss
            + self.config.offset_weight * offset_loss
            + self.config.hierarchy_consistency_weight * hierarchy_loss
        )
        return LossBreakdown(
            total=total,
            country=country_loss,
            region=region_loss,
            geocell=geocell_loss,
            geodesic=geodesic_loss,
            embedding=embedding_loss,
            offset=offset_loss,
            hierarchy=hierarchy_loss,
        )
