"""Fusion Head v0 modules."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class FusionModelConfig:
    structure_dim: int
    q_dim: int
    texture_dim: int = 0
    projected_structure_dim: int = 64
    projected_quality_dim: int = 16
    projected_texture_dim: int = 128
    hidden_dim: int = 128
    num_classes: int = 2
    dropout: float = 0.1


class FusionHeadV0(nn.Module):
    """Projection + concat + MLP classifier.

    The texture branch is optional so Stage 0 smoke training can start from
    structure/q vectors before CNN texture embeddings are cached.
    """

    def __init__(self, config: FusionModelConfig):
        super().__init__()
        self.config = config

        self.structure_projection = nn.Sequential(
            nn.Linear(int(config.structure_dim), int(config.projected_structure_dim)),
            nn.ReLU(inplace=True),
        )
        self.quality_projection = nn.Sequential(
            nn.Linear(int(config.q_dim), int(config.projected_quality_dim)),
            nn.ReLU(inplace=True),
        )

        if int(config.texture_dim) > 0:
            self.texture_projection: nn.Module | None = nn.Sequential(
                nn.LayerNorm(int(config.texture_dim)),
                nn.Linear(int(config.texture_dim), int(config.projected_texture_dim)),
                nn.ReLU(inplace=True),
            )
            texture_out = int(config.projected_texture_dim)
        else:
            self.texture_projection = None
            texture_out = 0

        fusion_dim = int(config.projected_structure_dim) + int(config.projected_quality_dim) + texture_out
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim, int(config.hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(config.dropout)),
            nn.Linear(int(config.hidden_dim), int(config.num_classes)),
        )

    def forward(
        self,
        structure_vector: torch.Tensor,
        q_vec: torch.Tensor,
        texture_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [
            self.structure_projection(structure_vector),
            self.quality_projection(q_vec),
        ]
        if self.texture_projection is not None:
            if texture_embedding is None:
                raise ValueError("texture_embedding is required when texture_dim > 0")
            parts.append(self.texture_projection(texture_embedding))
        fused = torch.cat(parts, dim=1)
        return self.fusion_mlp(fused)


@dataclass(frozen=True)
class FusionModelV02Config:
    structure_dim: int
    q_dim: int
    roi_texture_dim: int
    full_texture_dim: int
    projected_structure_dim: int = 64
    projected_quality_dim: int = 16
    projected_texture_dim: int = 128
    hidden_dim: int = 160
    num_classes: int = 2
    dropout: float = 0.1


class FusionHeadV02DualTexture(nn.Module):
    """MVP v0.2 head: ROI texture + full-image texture + q-conditioned gate."""

    def __init__(self, config: FusionModelV02Config):
        super().__init__()
        self.config = config
        self.structure_projection = nn.Sequential(
            nn.Linear(int(config.structure_dim), int(config.projected_structure_dim)),
            nn.ReLU(inplace=True),
        )
        self.quality_projection = nn.Sequential(
            nn.Linear(int(config.q_dim), int(config.projected_quality_dim)),
            nn.ReLU(inplace=True),
        )
        self.roi_texture_projection = nn.Sequential(
            nn.LayerNorm(int(config.roi_texture_dim)),
            nn.Linear(int(config.roi_texture_dim), int(config.projected_texture_dim)),
            nn.ReLU(inplace=True),
        )
        self.full_texture_projection = nn.Sequential(
            nn.LayerNorm(int(config.full_texture_dim)),
            nn.Linear(int(config.full_texture_dim), int(config.projected_texture_dim)),
            nn.ReLU(inplace=True),
        )
        self.roi_gate = nn.Sequential(
            nn.Linear(int(config.q_dim), 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        fusion_dim = (
            int(config.projected_structure_dim)
            + int(config.projected_quality_dim)
            + int(config.projected_texture_dim)
            + 1
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim, int(config.hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(config.dropout)),
            nn.Linear(int(config.hidden_dim), int(config.num_classes)),
        )

    def forward(
        self,
        structure_vector: torch.Tensor,
        q_vec: torch.Tensor,
        roi_texture_embedding: torch.Tensor,
        full_texture_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        roi_tex = self.roi_texture_projection(roi_texture_embedding)
        full_tex = self.full_texture_projection(full_texture_embedding)
        gate = self.roi_gate(q_vec)
        mixed_tex = gate * roi_tex + (1.0 - gate) * full_tex
        fused = torch.cat(
            [
                self.structure_projection(structure_vector),
                self.quality_projection(q_vec),
                mixed_tex,
                gate,
            ],
            dim=1,
        )
        return self.fusion_mlp(fused), gate
