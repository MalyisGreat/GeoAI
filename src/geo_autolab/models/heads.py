from __future__ import annotations

import torch
from torch import Tensor, nn


class MultiTaskGeoHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        country_classes: int,
        region_classes: int,
        geocell_classes: int,
        dropout: float,
        predict_country: bool,
        predict_region: bool,
        predict_geocell: bool,
        predict_uncertainty: bool,
        decode_topk: int,
        max_offset_norm: float,
        decode_confidence_threshold: float,
        decode_confidence_sharpness: float,
    ) -> None:
        super().__init__()
        self.predict_country = predict_country and country_classes > 1
        self.predict_region = predict_region and region_classes > 1
        self.predict_geocell = predict_geocell and geocell_classes > 0
        self.predict_uncertainty = predict_uncertainty
        self.decode_topk = max(1, decode_topk)
        self.max_offset_norm = max_offset_norm
        self.decode_confidence_threshold = decode_confidence_threshold
        self.decode_confidence_sharpness = decode_confidence_sharpness
        self.hidden_dim = hidden_dim

        self.pre_head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.embedding_head = nn.Linear(hidden_dim, embedding_dim)
        self.xyz_head = nn.Linear(hidden_dim, 3)

        self.country_head = nn.Linear(hidden_dim, country_classes) if self.predict_country else None
        self.country_embeddings = nn.Parameter(torch.randn(country_classes, hidden_dim) * 0.02) if self.predict_country else None

        region_input_dim = hidden_dim * (2 if self.predict_country else 1)
        self.region_context = nn.Sequential(
            nn.LayerNorm(region_input_dim),
            nn.Linear(region_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.region_head = nn.Linear(hidden_dim, region_classes) if self.predict_region else None
        self.region_embeddings = nn.Parameter(torch.randn(region_classes, hidden_dim) * 0.02) if self.predict_region else None

        geocell_input_dim = hidden_dim * (1 + int(self.predict_country) + int(self.predict_region))
        self.geocell_context = nn.Sequential(
            nn.LayerNorm(geocell_input_dim),
            nn.Linear(geocell_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.geocell_head = nn.Linear(hidden_dim, geocell_classes) if self.predict_geocell else None
        self.offset_head = nn.Linear(hidden_dim, 3) if self.predict_geocell else None
        self.uncertainty_head = nn.Linear(hidden_dim, 1) if self.predict_uncertainty else None

        self.register_buffer("geocell_centroids", torch.empty((0, 3), dtype=torch.float32), persistent=True)

    def set_geocell_centroids(self, centroids: Tensor) -> None:
        if centroids.ndim != 2 or centroids.shape[-1] != 3:
            raise ValueError(f"Expected centroids with shape [classes, 3], got {tuple(centroids.shape)}")
        self.geocell_centroids = nn.functional.normalize(centroids.detach().clone(), dim=-1)

    def _decode_geocell_conditioned_xyz(self, geocell_logits: Tensor, raw_offset: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        probs = geocell_logits.softmax(dim=-1)
        topk = min(self.decode_topk, probs.shape[-1])
        if topk < probs.shape[-1]:
            topk_probs, topk_idx = probs.topk(topk, dim=-1)
            topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            selected = self.geocell_centroids[topk_idx]
            base_centroid = (topk_probs.unsqueeze(-1) * selected).sum(dim=1)
        else:
            base_centroid = probs @ self.geocell_centroids
        base_centroid = nn.functional.normalize(base_centroid, dim=-1)

        bounded_offset = torch.tanh(raw_offset) * self.max_offset_norm
        local_offset = bounded_offset - torch.sum(bounded_offset * base_centroid, dim=-1, keepdim=True) * base_centroid
        decoded_xyz = nn.functional.normalize(base_centroid + local_offset, dim=-1)
        return decoded_xyz, base_centroid, local_offset, probs

    def _country_context(self, hidden: Tensor) -> tuple[Tensor | None, Tensor | None, Tensor]:
        if self.country_head is None or self.country_embeddings is None:
            return None, None, torch.zeros_like(hidden)
        country_logits = self.country_head(hidden)
        country_probs = country_logits.softmax(dim=-1)
        country_context = country_probs @ self.country_embeddings
        return country_logits, country_probs, country_context

    def _region_context(self, hidden: Tensor, country_context: Tensor) -> tuple[Tensor | None, Tensor | None, Tensor]:
        region_inputs = [hidden]
        if self.predict_country:
            region_inputs.append(country_context)
        region_features = self.region_context(torch.cat(region_inputs, dim=-1))
        if self.region_head is None or self.region_embeddings is None:
            return None, None, torch.zeros_like(region_features)
        region_logits = self.region_head(region_features)
        region_probs = region_logits.softmax(dim=-1)
        region_context = region_probs @ self.region_embeddings
        return region_logits, region_probs, region_context

    def forward(self, features: Tensor) -> dict[str, Tensor | None]:
        hidden = self.pre_head(features)
        embedding = nn.functional.normalize(self.embedding_head(hidden), dim=-1)
        fallback_xyz = nn.functional.normalize(self.xyz_head(hidden), dim=-1)

        country_logits, country_probs, country_context = self._country_context(hidden)
        region_logits, region_probs, region_context = self._region_context(hidden, country_context)

        geocell_inputs = [hidden]
        if self.predict_country:
            geocell_inputs.append(country_context)
        if self.predict_region:
            geocell_inputs.append(region_context)
        geocell_features = self.geocell_context(torch.cat(geocell_inputs, dim=-1))

        geocell_logits = self.geocell_head(geocell_features) if self.geocell_head is not None else None
        local_offset = None
        base_centroid = None
        geocell_probs = None
        decode_mix = None
        unit_xyz = fallback_xyz
        if (
            geocell_logits is not None
            and self.offset_head is not None
            and self.geocell_centroids.numel() > 0
            and self.geocell_centroids.shape[0] == geocell_logits.shape[-1]
        ):
            decoded_xyz, base_centroid, local_offset, geocell_probs = self._decode_geocell_conditioned_xyz(
                geocell_logits=geocell_logits,
                raw_offset=self.offset_head(geocell_features),
            )
            decode_confidence = geocell_probs.max(dim=-1, keepdim=True).values
            decode_mix = torch.sigmoid(
                (decode_confidence - self.decode_confidence_threshold) * self.decode_confidence_sharpness
            )
            unit_xyz = nn.functional.normalize(
                (1.0 - decode_mix) * fallback_xyz + decode_mix * decoded_xyz,
                dim=-1,
            )

        uncertainty = None
        if self.uncertainty_head is not None:
            uncertainty = nn.functional.softplus(self.uncertainty_head(geocell_features)) + 1e-3

        return {
            "embedding": embedding,
            "unit_xyz": unit_xyz,
            "fallback_unit_xyz": fallback_xyz,
            "country_logits": country_logits,
            "country_probs": country_probs,
            "region_logits": region_logits,
            "region_probs": region_probs,
            "geocell_logits": geocell_logits,
            "geocell_probs": geocell_probs,
            "base_centroid": base_centroid,
            "local_offset": local_offset,
            "decode_mix": decode_mix,
            "uncertainty": uncertainty,
        }
