"""Chain-free Joint Parent/Fine validity and action gate.

Evidence providers run independently and populate four packets.  This module
does not call legacy v153/v210/v214 stages and does not read artifacts.  It
only interprets the packets supplied to one forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


JOINT_STATES = ("both_valid", "parent_only", "fine_only", "neither")
JOINT_ACTIONS = (
    "accept_both",
    "repair_parent",
    "repair_fine",
    "reobserve_parent",
    "reobserve_fine",
    "reobserve_both",
    "abstain",
)


@dataclass(slots=True)
class JointGateInputV216:
    """Parallel evidence packets consumed by the v216 gate.

    All feature tensors use shape ``[batch, feature_dim]``.  Truth-derived
    correctness fields intentionally do not exist in this runtime contract.
    """

    sample_keys: list[str]
    parent_features: torch.Tensor
    fine_features: torch.Tensor
    cross_features: torch.Tensor
    observation_features: torch.Tensor
    metadata: dict[str, Any] | None = None

    def validate(self, spec: "JointGateSpecV216") -> None:
        packets = {
            "parent_features": (self.parent_features, spec.parent_dim),
            "fine_features": (self.fine_features, spec.fine_dim),
            "cross_features": (self.cross_features, spec.cross_dim),
            "observation_features": (self.observation_features, spec.observation_dim),
        }
        batch_size = len(self.sample_keys)
        for name, (value, expected_dim) in packets.items():
            if value.ndim != 2:
                raise ValueError(f"{name} must have shape [batch, feature_dim], got {tuple(value.shape)}")
            if value.shape[0] != batch_size:
                raise ValueError(f"{name} batch {value.shape[0]} != sample key count {batch_size}")
            if value.shape[1] != expected_dim:
                raise ValueError(f"{name} dim {value.shape[1]} != expected {expected_dim}")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")


@dataclass(frozen=True, slots=True)
class JointGateSpecV216:
    parent_dim: int = 5
    fine_dim: int = 7
    cross_dim: int = 6
    observation_dim: int = 4
    parent_hidden_dim: int = 16
    parent_state_dim: int = 8
    fine_hidden_dim: int = 20
    fine_state_dim: int = 10
    relation_projection_dim: int = 8
    relation_hidden_dim: int = 32
    relation_state_dim: int = 16
    action_hidden_dim: int = 32
    dropout: float = 0.1


@dataclass(slots=True)
class JointGateOutputV216:
    parent_validity_logit: torch.Tensor
    fine_validity_logit: torch.Tensor
    parent_validity: torch.Tensor
    fine_validity: torch.Tensor
    state_logits: torch.Tensor
    state_probabilities: torch.Tensor
    action_logits: torch.Tensor
    action_probabilities: torch.Tensor
    relation_state: torch.Tensor


class _StateEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class JointGateV216(nn.Module):
    """Interpret Parent/Fine evidence jointly without imposing a tree route."""

    def __init__(self, spec: JointGateSpecV216 = JointGateSpecV216()) -> None:
        super().__init__()
        self.spec = spec
        self.parent_encoder = _StateEncoder(spec.parent_dim, spec.parent_hidden_dim, spec.parent_state_dim)
        self.fine_encoder = _StateEncoder(spec.fine_dim, spec.fine_hidden_dim, spec.fine_state_dim)
        self.parent_relation_projection = nn.Linear(spec.parent_state_dim, spec.relation_projection_dim)
        self.fine_relation_projection = nn.Linear(spec.fine_state_dim, spec.relation_projection_dim)
        relation_input_dim = spec.relation_projection_dim * 4 + spec.cross_dim
        self.relation_encoder = nn.Sequential(
            nn.Linear(relation_input_dim, spec.relation_hidden_dim),
            nn.LayerNorm(spec.relation_hidden_dim),
            nn.GELU(),
            nn.Dropout(spec.dropout),
            nn.Linear(spec.relation_hidden_dim, spec.relation_state_dim),
        )
        self.parent_validity_head = nn.Linear(spec.relation_state_dim, 1)
        self.fine_validity_head = nn.Linear(spec.relation_state_dim, 1)
        self.state_head = nn.Linear(spec.relation_state_dim, len(JOINT_STATES))
        action_input_dim = spec.relation_state_dim + 2 + len(JOINT_STATES) + spec.observation_dim
        self.action_head = nn.Sequential(
            nn.Linear(action_input_dim, spec.action_hidden_dim),
            nn.LayerNorm(spec.action_hidden_dim),
            nn.GELU(),
            nn.Dropout(spec.dropout),
            nn.Linear(spec.action_hidden_dim, len(JOINT_ACTIONS)),
        )

    def forward(self, batch: JointGateInputV216) -> JointGateOutputV216:
        batch.validate(self.spec)
        parent_state = self.parent_encoder(batch.parent_features)
        fine_state = self.fine_encoder(batch.fine_features)
        parent_relation = self.parent_relation_projection(parent_state)
        fine_relation = self.fine_relation_projection(fine_state)
        relation_input = torch.cat(
            [
                parent_relation,
                fine_relation,
                parent_relation - fine_relation,
                parent_relation * fine_relation,
                batch.cross_features,
            ],
            dim=1,
        )
        relation_state = self.relation_encoder(relation_input)
        parent_logit = self.parent_validity_head(relation_state).squeeze(1)
        fine_logit = self.fine_validity_head(relation_state).squeeze(1)
        parent_validity = torch.sigmoid(parent_logit)
        fine_validity = torch.sigmoid(fine_logit)
        state_logits = self.state_head(relation_state)
        state_probabilities = F.softmax(state_logits, dim=1)
        action_input = torch.cat(
            [
                relation_state,
                parent_validity[:, None],
                fine_validity[:, None],
                state_probabilities,
                batch.observation_features,
            ],
            dim=1,
        )
        action_logits = self.action_head(action_input)
        return JointGateOutputV216(
            parent_validity_logit=parent_logit,
            fine_validity_logit=fine_logit,
            parent_validity=parent_validity,
            fine_validity=fine_validity,
            state_logits=state_logits,
            state_probabilities=state_probabilities,
            action_logits=action_logits,
            action_probabilities=F.softmax(action_logits, dim=1),
            relation_state=relation_state,
        )


def joint_gate_loss_v216(
    output: JointGateOutputV216,
    *,
    parent_valid_target: torch.Tensor,
    fine_valid_target: torch.Tensor,
    state_target: torch.Tensor,
    action_target: torch.Tensor | None = None,
    parent_weight: float = 1.0,
    fine_weight: float = 1.0,
    state_weight: float = 0.2,
    action_weight: float = 0.0,
    parent_sample_weight: torch.Tensor | None = None,
    fine_sample_weight: torch.Tensor | None = None,
    state_class_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute staged v216 losses; action learning is disabled by default."""

    parent_items = F.binary_cross_entropy_with_logits(
        output.parent_validity_logit,
        parent_valid_target.float(),
        reduction="none",
    )
    fine_items = F.binary_cross_entropy_with_logits(
        output.fine_validity_logit,
        fine_valid_target.float(),
        reduction="none",
    )
    if parent_sample_weight is not None:
        parent_items = parent_items * parent_sample_weight
    if fine_sample_weight is not None:
        fine_items = fine_items * fine_sample_weight
    parent_loss = parent_items.mean()
    fine_loss = fine_items.mean()
    state_loss = F.cross_entropy(output.state_logits, state_target.long(), weight=state_class_weight)
    action_loss = output.action_logits.sum() * 0.0
    if action_target is not None and action_weight > 0.0:
        action_loss = F.cross_entropy(output.action_logits, action_target.long())
    total = (
        parent_weight * parent_loss
        + fine_weight * fine_loss
        + state_weight * state_loss
        + action_weight * action_loss
    )
    return total, {
        "parent_validity": parent_loss,
        "fine_validity": fine_loss,
        "joint_state": state_loss,
        "action": action_loss,
    }
