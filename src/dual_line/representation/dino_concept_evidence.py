"""Independent Parent/Fine evidence projectors for frozen DINO tokens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class ConceptLayout:
    """Class graph supplied by configuration rather than hard-coded labels."""

    fine_classes: tuple[str, ...]
    parent_classes: tuple[str, ...]
    fine_to_parent_index: tuple[int, ...]

    @classmethod
    def from_parent_map(
        cls,
        fine_classes: Sequence[str],
        parent_map: Mapping[str, str],
    ) -> "ConceptLayout":
        fine = tuple(str(x) for x in fine_classes)
        missing = [name for name in fine if name not in parent_map]
        if missing:
            raise ValueError(f"parent map is missing fine classes: {missing}")
        parents: list[str] = []
        indices: list[int] = []
        for name in fine:
            parent = str(parent_map[name])
            if parent not in parents:
                parents.append(parent)
            indices.append(parents.index(parent))
        return cls(
            fine_classes=fine,
            parent_classes=tuple(parents),
            fine_to_parent_index=tuple(indices),
        )

    def parent_targets(self, fine_targets: torch.Tensor) -> torch.Tensor:
        mapping = torch.as_tensor(
            self.fine_to_parent_index,
            dtype=torch.long,
            device=fine_targets.device,
        )
        return mapping.index_select(0, fine_targets.long())


@dataclass(frozen=True, slots=True)
class DinoConceptEvidenceOutput:
    parent_logits: torch.Tensor
    fine_logits: torch.Tensor
    parent_cls_logits: torch.Tensor
    fine_cls_logits: torch.Tensor
    parent_tile_logits: torch.Tensor
    fine_tile_logits: torch.Tensor
    parent_tile_pooled_logits: torch.Tensor
    fine_tile_pooled_logits: torch.Tensor
    parent_tile_evidence: torch.Tensor
    fine_tile_evidence: torch.Tensor
    parent_spatial_attention: torch.Tensor
    fine_spatial_attention: torch.Tensor
    valid_tile_mask: torch.Tensor


class _ClsHead(nn.Module):
    def __init__(self, dim: int, classes: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DinoConceptEvidenceProjector(nn.Module):
    """Learn separate common-identity and fine-contrast evidence maps.

    DINO stays frozen outside this module.  Parent and Fine projectors share
    input tokens but do not share trainable heads, preserving their different
    task-specific representations.
    """

    def __init__(
        self,
        *,
        embedding_dim: int,
        parent_classes: int,
        fine_classes: int,
        hidden_dim: int = 160,
        dropout: float = 0.10,
        attention_temperature: float = 0.75,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.parent_classes = int(parent_classes)
        self.fine_classes = int(fine_classes)
        self.attention_temperature = float(attention_temperature)
        self.parent_cls_head = _ClsHead(self.embedding_dim, self.parent_classes, hidden_dim, dropout)
        self.fine_cls_head = _ClsHead(self.embedding_dim, self.fine_classes, hidden_dim, dropout)
        self.parent_tile_head = nn.Linear(self.embedding_dim, self.parent_classes)
        self.fine_tile_head = nn.Linear(self.embedding_dim, self.fine_classes)
        self.parent_fusion_logit = nn.Parameter(torch.tensor(0.0))
        self.fine_fusion_logit = nn.Parameter(torch.tensor(0.0))

    def _pool_tiles(
        self,
        logits: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        masked = logits.masked_fill(~valid_mask.unsqueeze(-1), -1.0e4)
        attention = F.softmax(masked.transpose(1, 2) / self.attention_temperature, dim=-1)
        pooled = torch.einsum("bct,btc->bc", attention, logits)
        return pooled, attention

    def forward(
        self,
        cls_token: torch.Tensor,
        tile_tokens: torch.Tensor,
        *,
        valid_tile_mask: torch.Tensor | None = None,
    ) -> DinoConceptEvidenceOutput:
        if cls_token.ndim != 2:
            raise ValueError(f"cls_token must be [B,D], got {tuple(cls_token.shape)}")
        if tile_tokens.ndim != 3:
            raise ValueError(f"tile_tokens must be [B,T,D], got {tuple(tile_tokens.shape)}")
        if cls_token.shape[0] != tile_tokens.shape[0] or cls_token.shape[-1] != tile_tokens.shape[-1]:
            raise ValueError("CLS and tile token batch/embedding dimensions differ")
        if valid_tile_mask is None:
            valid_tile_mask = torch.ones(
                tile_tokens.shape[:2],
                dtype=torch.bool,
                device=tile_tokens.device,
            )
        else:
            valid_tile_mask = valid_tile_mask.to(tile_tokens.device, dtype=torch.bool)
        if valid_tile_mask.shape != tile_tokens.shape[:2]:
            raise ValueError("valid_tile_mask must be [B,T]")
        if not bool(valid_tile_mask.any(dim=1).all()):
            raise ValueError("every sample must keep at least one valid tile")

        parent_cls = self.parent_cls_head(cls_token)
        fine_cls = self.fine_cls_head(cls_token)
        parent_tile = self.parent_tile_head(tile_tokens)
        fine_tile = self.fine_tile_head(tile_tokens)
        parent_pooled, parent_attention = self._pool_tiles(parent_tile, valid_tile_mask)
        fine_pooled, fine_attention = self._pool_tiles(fine_tile, valid_tile_mask)
        parent_alpha = torch.sigmoid(self.parent_fusion_logit)
        fine_alpha = torch.sigmoid(self.fine_fusion_logit)
        parent_logits = parent_alpha * parent_cls + (1.0 - parent_alpha) * parent_pooled
        fine_logits = fine_alpha * fine_cls + (1.0 - fine_alpha) * fine_pooled
        return DinoConceptEvidenceOutput(
            parent_logits=parent_logits,
            fine_logits=fine_logits,
            parent_cls_logits=parent_cls,
            fine_cls_logits=fine_cls,
            parent_tile_logits=parent_tile,
            fine_tile_logits=fine_tile,
            parent_tile_pooled_logits=parent_pooled,
            fine_tile_pooled_logits=fine_pooled,
            parent_tile_evidence=F.softmax(parent_tile, dim=-1),
            fine_tile_evidence=F.softmax(fine_tile, dim=-1),
            parent_spatial_attention=parent_attention,
            fine_spatial_attention=fine_attention,
            valid_tile_mask=valid_tile_mask,
        )


def symmetric_kl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    log_a = F.log_softmax(a, dim=-1)
    log_b = F.log_softmax(b, dim=-1)
    prob_a = log_a.exp()
    prob_b = log_b.exp()
    return 0.5 * (
        F.kl_div(log_a, prob_b, reduction="batchmean")
        + F.kl_div(log_b, prob_a, reduction="batchmean")
    )


def true_class_attention_concentration(
    attention: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    selected = attention[torch.arange(len(targets), device=targets.device), targets.long()]
    return selected.max(dim=1).values.mean()
