from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torch.nn import functional as F

from .backbones import build_backbone


class ExpertBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ProbabilisticGlobeHead(nn.Module):
    def __init__(self, dim: int, components: int) -> None:
        super().__init__()
        self.components = components
        self.weight_head = nn.Linear(dim, components)
        self.mean_head = nn.Linear(dim, components * 3)
        self.kappa_head = nn.Linear(dim, components)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mixture_logits = self.weight_head(x)
        means = self.mean_head(x).view(x.shape[0], self.components, 3)
        means = F.normalize(means, dim=-1)
        kappa = F.softplus(self.kappa_head(x)) + 1e-3
        weights = mixture_logits.softmax(dim=-1)
        expected_unit = F.normalize((weights.unsqueeze(-1) * means).sum(dim=1), dim=-1)
        return {
            "posterior_logits": mixture_logits,
            "posterior_means": means,
            "posterior_kappa": kappa,
            "posterior_expected_unit": expected_unit,
        }


class GeoLocator(nn.Module):
    def __init__(
        self,
        *,
        backbone_name: str,
        fallback_backbone: str,
        pretrained: bool,
        hidden_dim: int,
        embedding_dim: int,
        dropout: float,
        num_coarse_classes: int,
        num_fine_classes: int,
        num_experts: int = 4,
        posterior_components: int = 8,
        retrieval_temperature: float = 0.07,
        aux_head_dims: dict[str, int] | None = None,
        numeric_head_names: Iterable[str] | None = None,
    ) -> None:
        super().__init__()
        self.backbone = build_backbone(backbone_name, fallback_backbone, pretrained)
        self.shared_projection = nn.Sequential(
            nn.Linear(self.backbone.out_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.router_head = nn.Linear(embedding_dim, num_coarse_classes)
        self.expert_gate = nn.Linear(embedding_dim, num_experts)
        self.experts = nn.ModuleList(
            [ExpertBlock(embedding_dim, hidden_dim, dropout) for _ in range(num_experts)]
        )
        self.fine_head = nn.Linear(embedding_dim, num_fine_classes)
        self.regression_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )
        self.retrieval_projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.cell_prototypes = nn.Embedding(num_fine_classes, embedding_dim)
        self.probabilistic_head = ProbabilisticGlobeHead(embedding_dim, posterior_components)
        self.retrieval_temperature = retrieval_temperature

        self.clue_projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.aux_head_dims = aux_head_dims or {}
        self.numeric_head_names = list(numeric_head_names or [])
        self.clue_logits = nn.ModuleDict(
            {name: nn.Linear(hidden_dim, size) for name, size in self.aux_head_dims.items()}
        )
        self.clue_numeric = nn.ModuleDict(
            {name: nn.Linear(hidden_dim, 1) for name in self.numeric_head_names}
        )

    def _build_clue_vector(
        self,
        categorical_outputs: dict[str, torch.Tensor],
        numeric_outputs: dict[str, torch.Tensor],
    ) -> torch.Tensor | None:
        parts: list[torch.Tensor] = []
        for name in sorted(categorical_outputs):
            parts.append(categorical_outputs[name].softmax(dim=-1))
        for name in sorted(numeric_outputs):
            parts.append(torch.tanh(numeric_outputs[name]).unsqueeze(-1))
        if not parts:
            return None
        return torch.cat(parts, dim=-1)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(images)
        shared_embedding = F.normalize(self.shared_projection(features), dim=-1)
        coarse_logits = self.router_head(shared_embedding)

        expert_weights = self.expert_gate(shared_embedding).softmax(dim=-1)
        expert_outputs = torch.stack([expert(shared_embedding) for expert in self.experts], dim=1)
        embedding = F.normalize((expert_weights.unsqueeze(-1) * expert_outputs).sum(dim=1), dim=-1)

        fine_logits = self.fine_head(embedding)
        retrieval_embedding = F.normalize(self.retrieval_projection(embedding), dim=-1)
        prototype_matrix = F.normalize(self.cell_prototypes.weight, dim=-1)
        retrieval_logits = (retrieval_embedding @ prototype_matrix.T) / self.retrieval_temperature

        posterior = self.probabilistic_head(embedding)
        regressed_unit = F.normalize(self.regression_head(embedding), dim=-1)
        coord_unit = F.normalize(
            0.5 * posterior["posterior_expected_unit"] + 0.5 * regressed_unit,
            dim=-1,
        )

        clue_context = self.clue_projection(embedding)
        clue_logits = {name: head(clue_context) for name, head in self.clue_logits.items()}
        clue_numeric = {
            name: head(clue_context).squeeze(-1) for name, head in self.clue_numeric.items()
        }
        clue_vector = self._build_clue_vector(clue_logits, clue_numeric)

        return {
            "shared_embedding": shared_embedding,
            "embedding": embedding,
            "coarse_logits": coarse_logits,
            "fine_logits": fine_logits,
            "retrieval_embedding": retrieval_embedding,
            "retrieval_logits": retrieval_logits,
            "coord_unit": coord_unit,
            "regressed_unit": regressed_unit,
            "expert_weights": expert_weights,
            "clue_vector": clue_vector
            if clue_vector is not None
            else embedding.new_zeros((embedding.shape[0], 0)),
            "clue_logits": clue_logits,
            "clue_numeric": clue_numeric,
            **posterior,
        }
