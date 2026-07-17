"""DINO spatial evidence aligned with the 4x4 observer relation grid.

This module is intentionally label-free and decision-free.  It compares the
semantic relation exposed by DINO patch tokens with the geometric continuity
exposed by the legacy tile/wave observer.  A later gate may interpret the
result, but this adapter never chooses a class or approves a transition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dual_line.backbone_adapters import DinoSpatialFeatures
from src.dual_line.runtime_types import RelationBatch


DINO_OBSERVER_GATE_FEATURE_NAMES = (
    "dino_cls_tile_mean",
    "dino_cls_tile_max",
    "dino_cls_tile_std",
    "dino_semantic_local_mean",
    "dino_semantic_local_max",
    "dino_semantic_local_std",
    "dino_semantic_local_density",
    "dino_semantic_nonlocal_mean",
    "dino_semantic_local_nonlocal_gap",
    "wave_local_mean",
    "wave_local_std",
    "wave_directional_asymmetry_mean",
    "dino_wave_agreement_mean",
    "dino_wave_agreement_max",
    "dino_only_conflict_mean",
    "wave_only_conflict_mean",
    "dino_wave_abs_gap_mean",
)


@dataclass(frozen=True, slots=True)
class DinoObserverRelationConfig:
    grid: int = 4
    semantic_threshold: float = 0.70
    eps: float = 1.0e-6


@dataclass(frozen=True, slots=True)
class DinoObserverRelationOutput:
    """Relation tensors and compact label-free features for a later gate."""

    cls_tile_affinity: torch.Tensor
    semantic_affinity: torch.Tensor
    local_neighbor_mask: torch.Tensor
    boundary_tile_mask: torch.Tensor
    wave_affinity: torch.Tensor | None
    wave_directional_asymmetry: torch.Tensor | None
    dino_wave_agreement: torch.Tensor | None
    dino_only_conflict: torch.Tensor | None
    wave_only_conflict: torch.Tensor | None
    gate_features: torch.Tensor
    gate_feature_names: tuple[str, ...]
    metadata: dict[str, Any]


def finite_neighbor_mask(grid: int, *, device: torch.device | None = None) -> torch.Tensor:
    """Return a finite-image 8-neighbor mask without periodic wrapping."""

    grid = int(grid)
    if grid <= 0:
        raise ValueError(f"grid must be positive, got {grid}")
    tile_count = grid * grid
    mask = torch.zeros((tile_count, tile_count), dtype=torch.bool, device=device)
    for src in range(tile_count):
        src_r, src_c = divmod(src, grid)
        for dst in range(tile_count):
            if src == dst:
                continue
            dst_r, dst_c = divmod(dst, grid)
            if max(abs(src_r - dst_r), abs(src_c - dst_c)) == 1:
                mask[src, dst] = True
    return mask


def boundary_tile_mask(grid: int, *, device: torch.device | None = None) -> torch.Tensor:
    """Mark tiles touching the finite image boundary."""

    grid = int(grid)
    out = torch.zeros(grid * grid, dtype=torch.bool, device=device)
    for tile in range(grid * grid):
        row, col = divmod(tile, grid)
        out[tile] = row in {0, grid - 1} or col in {0, grid - 1}
    return out


def _feature_index(feature_names: Sequence[str], name: str) -> int:
    try:
        return [str(x) for x in feature_names].index(name)
    except ValueError as exc:
        raise ValueError(f"wave relation feature is required: {name}") from exc


def project_wave_affinity(
    wave_relations: torch.Tensor,
    feature_names: Sequence[str],
    *,
    eps: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project legacy directional wave features into a soft continuity map.

    Coverage says whether one observer can see another tile.  Edge delta says
    whether that observation remains geometrically continuous.  The projection
    is normalized per image and does not use class labels or correctness.
    """

    if wave_relations.ndim != 4:
        raise ValueError(
            "wave_relations must be [B,T,T,C], "
            f"got shape={tuple(wave_relations.shape)}"
        )
    coverage_i = _feature_index(feature_names, "mutual_coverage_min")
    edge_delta_i = _feature_index(feature_names, "mutual_edge_delta_abs_mean")
    coverage = wave_relations[..., coverage_i].float().clamp_min(0.0)
    edge_delta = wave_relations[..., edge_delta_i].float().abs()

    coverage_max = coverage.amax(dim=(-2, -1), keepdim=True).clamp_min(float(eps))
    coverage_support = (coverage / coverage_max).clamp(0.0, 1.0)
    edge_scale = edge_delta.mean(dim=(-2, -1), keepdim=True).clamp_min(float(eps))
    edge_continuity = torch.exp(-edge_delta / edge_scale)
    directional = (coverage_support * edge_continuity).clamp(0.0, 1.0)
    symmetric = 0.5 * (directional + directional.transpose(-1, -2))
    asymmetry = (directional - directional.transpose(-1, -2)).abs()
    return symmetric, asymmetry


def wave_affinity_from_relation_batch(
    batch: RelationBatch,
    *,
    device: torch.device,
    eps: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a runtime v05 RelationBatch to DINO-aligned wave affinity."""

    if batch.wave_relations is None:
        raise ValueError("RelationBatch.wave_relations is required")
    feature_names = [str(x) for x in batch.metadata.get("feature_names", [])]
    relation = torch.as_tensor(
        np.asarray(batch.wave_relations, dtype=np.float32),
        device=device,
    )
    return project_wave_affinity(relation, feature_names, eps=eps)


def _masked_values(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.unsqueeze(0).expand(values.shape[0], -1, -1)
    return values.masked_select(expanded).reshape(values.shape[0], -1)


def _stats(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return values.mean(dim=1), values.max(dim=1).values, values.std(dim=1, unbiased=False)


class DinoObserverRelationAdapter(nn.Module):
    """Build semantic/geometric relation evidence for a DINO-aware gate."""

    def __init__(self, config: DinoObserverRelationConfig | None = None) -> None:
        super().__init__()
        self.config = config or DinoObserverRelationConfig()
        local = finite_neighbor_mask(self.config.grid)
        boundary = boundary_tile_mask(self.config.grid)
        self.register_buffer("local_neighbor_mask", local, persistent=False)
        self.register_buffer("boundary_tile_mask", boundary, persistent=False)

    def forward(
        self,
        spatial: DinoSpatialFeatures,
        *,
        wave_affinity: torch.Tensor | None = None,
        wave_directional_asymmetry: torch.Tensor | None = None,
    ) -> DinoObserverRelationOutput:
        tile_tokens = spatial.tile_tokens
        cls_token = spatial.cls_token
        expected_grid = (self.config.grid, self.config.grid)
        if spatial.tile_grid != expected_grid:
            raise ValueError(
                f"observer tile grid mismatch: expected={expected_grid} got={spatial.tile_grid}"
            )
        tile_count = self.config.grid * self.config.grid
        if tile_tokens.ndim != 3 or int(tile_tokens.shape[1]) != tile_count:
            raise ValueError(
                f"tile_tokens must be [B,{tile_count},D], got {tuple(tile_tokens.shape)}"
            )

        tile_norm = F.normalize(tile_tokens.float(), dim=-1, eps=self.config.eps)
        cls_norm = F.normalize(cls_token.float(), dim=-1, eps=self.config.eps)
        semantic_cosine = torch.einsum("btd,bsd->bts", tile_norm, tile_norm)
        semantic_affinity = ((semantic_cosine + 1.0) * 0.5).clamp(0.0, 1.0)
        cls_tile_cosine = torch.einsum("bd,btd->bt", cls_norm, tile_norm)
        cls_tile_affinity = ((cls_tile_cosine + 1.0) * 0.5).clamp(0.0, 1.0)

        local_mask = self.local_neighbor_mask.to(tile_tokens.device)
        diagonal = torch.eye(tile_count, dtype=torch.bool, device=tile_tokens.device)
        nonlocal_mask = ~(local_mask | diagonal)
        semantic_local = _masked_values(semantic_affinity, local_mask)
        semantic_nonlocal = _masked_values(semantic_affinity, nonlocal_mask)
        sem_mean, sem_max, sem_std = _stats(semantic_local)
        cls_mean, cls_max, cls_std = _stats(cls_tile_affinity)
        semantic_density = (semantic_local >= float(self.config.semantic_threshold)).float().mean(dim=1)
        nonlocal_mean = semantic_nonlocal.mean(dim=1)

        batch_size = tile_tokens.shape[0]
        zeros = torch.zeros(batch_size, device=tile_tokens.device, dtype=torch.float32)
        agreement = None
        dino_only = None
        wave_only = None
        if wave_affinity is not None:
            wave_affinity = wave_affinity.to(tile_tokens.device, dtype=torch.float32)
            if wave_affinity.shape != semantic_affinity.shape:
                raise ValueError(
                    "wave_affinity must match semantic relation shape: "
                    f"expected={tuple(semantic_affinity.shape)} got={tuple(wave_affinity.shape)}"
                )
            wave_affinity = wave_affinity.clamp(0.0, 1.0)
            agreement = semantic_affinity * wave_affinity
            dino_only = semantic_affinity * (1.0 - wave_affinity)
            wave_only = wave_affinity * (1.0 - semantic_affinity)
            wave_local = _masked_values(wave_affinity, local_mask)
            wave_mean, _, wave_std = _stats(wave_local)
            agreement_local = _masked_values(agreement, local_mask)
            agreement_mean, agreement_max, _ = _stats(agreement_local)
            dino_only_mean = _masked_values(dino_only, local_mask).mean(dim=1)
            wave_only_mean = _masked_values(wave_only, local_mask).mean(dim=1)
            abs_gap_mean = _masked_values((semantic_affinity - wave_affinity).abs(), local_mask).mean(dim=1)
        else:
            wave_mean = wave_std = agreement_mean = agreement_max = zeros
            dino_only_mean = wave_only_mean = abs_gap_mean = zeros

        if wave_directional_asymmetry is not None:
            wave_directional_asymmetry = wave_directional_asymmetry.to(
                tile_tokens.device,
                dtype=torch.float32,
            )
            if wave_directional_asymmetry.shape != semantic_affinity.shape:
                raise ValueError("wave_directional_asymmetry must match semantic relation shape")
            asymmetry_mean = _masked_values(wave_directional_asymmetry, local_mask).mean(dim=1)
        else:
            asymmetry_mean = zeros

        gate_features = torch.stack(
            [
                cls_mean,
                cls_max,
                cls_std,
                sem_mean,
                sem_max,
                sem_std,
                semantic_density,
                nonlocal_mean,
                sem_mean - nonlocal_mean,
                wave_mean,
                wave_std,
                asymmetry_mean,
                agreement_mean,
                agreement_max,
                dino_only_mean,
                wave_only_mean,
                abs_gap_mean,
            ],
            dim=1,
        )
        return DinoObserverRelationOutput(
            cls_tile_affinity=cls_tile_affinity,
            semantic_affinity=semantic_affinity,
            local_neighbor_mask=local_mask,
            boundary_tile_mask=self.boundary_tile_mask.to(tile_tokens.device),
            wave_affinity=wave_affinity,
            wave_directional_asymmetry=wave_directional_asymmetry,
            dino_wave_agreement=agreement,
            dino_only_conflict=dino_only,
            wave_only_conflict=wave_only,
            gate_features=gate_features,
            gate_feature_names=DINO_OBSERVER_GATE_FEATURE_NAMES,
            metadata={
                "mode": "dino_observer_relation",
                "grid": int(self.config.grid),
                "finite_boundary": True,
                "periodic_wrap": False,
                "semantic_threshold": float(self.config.semantic_threshold),
                "wave_available": wave_affinity is not None,
            },
        )
