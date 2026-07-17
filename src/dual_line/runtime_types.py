"""Common runtime batch contracts for the modular Dual-Line pipeline.

These dataclasses are intentionally lightweight.  They describe the data that
flows between future modules without forcing the existing v153 tool chain to be
rewritten all at once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


ArrayLike = Any


@dataclass(slots=True)
class ScanBatch:
    """Observed image state produced by the input/observation layer."""

    sample_keys: list[str]
    full_images: ArrayLike | None = None
    roi_images: ArrayLike | None = None
    tile_images: ArrayLike | None = None
    tile_positions: ArrayLike | None = None
    observer_features: ArrayLike | None = None
    valid_mask: ArrayLike | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RepresentationBatch:
    """Backbone and structure embeddings for a ScanBatch."""

    sample_keys: list[str]
    full_embedding: ArrayLike | None = None
    roi_embedding: ArrayLike | None = None
    tile_embedding: ArrayLike | None = None
    structure_embedding: ArrayLike | None = None
    observer_embedding: ArrayLike | None = None
    masks: dict[str, ArrayLike] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RelationBatch:
    """Pairwise and flow evidence built from representations."""

    sample_keys: list[str]
    texture_relations: ArrayLike | None = None
    object_relations: ArrayLike | None = None
    wave_relations: ArrayLike | None = None
    observer_relations: ArrayLike | None = None
    object_flow_relations: ArrayLike | None = None
    masks: dict[str, ArrayLike] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BBoxCandidateScoreBatch:
    """BBox/view candidate scores produced before evidence bundling."""

    sample_keys: list[str]
    bbox_keys: ArrayLike
    class_names: list[str]
    probs: ArrayLike
    view_label: ArrayLike
    view_conf: ArrayLike
    view_margin: ArrayLike
    bbox_features: ArrayLike
    bbox_feature_names: list[str]
    selected_class: ArrayLike | None = None
    selected_score: ArrayLike | None = None
    true_label: ArrayLike | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CandidateBatch:
    """Alternative predictions and evidence bundles before gate approval."""

    sample_keys: list[str]
    selected_class: ArrayLike | None = None
    selected_score: ArrayLike | None = None
    candidate_classes: ArrayLike | None = None
    candidate_scores: ArrayLike | None = None
    candidate_features: ArrayLike | None = None
    support_features: ArrayLike | None = None
    warning_features: ArrayLike | None = None
    candidate_mask: ArrayLike | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvidenceBatch:
    """Per-candidate evidence before selector scoring.

    This sits between RelationBatch and CandidateBatch.  It keeps the evidence
    matrix as a runtime tensor-like contract while allowing the current v153
    builder to keep using DataFrames internally during the migration.
    """

    sample_keys: list[str]
    evidence_classes: ArrayLike
    evidence_scores: ArrayLike | None = None
    evidence_features: ArrayLike | None = None
    evidence_feature_names: list[str] = field(default_factory=list)
    selected_class: ArrayLike | None = None
    selected_score: ArrayLike | None = None
    evidence_mask: ArrayLike | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ConceptEvidenceBatch:
    """Concept-graph evidence used by automatic gates."""

    sample_keys: list[str]
    support: ArrayLike | None = None
    warning: ArrayLike | None = None
    conflict: ArrayLike | None = None
    context: ArrayLike | None = None
    risk: ArrayLike | None = None
    sibling_relation: ArrayLike | None = None
    parent_relation: ArrayLike | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GateOutput:
    """Final decision output from a manual or automatic gate."""

    sample_keys: list[str]
    final_prediction: ArrayLike
    selected_prediction: ArrayLike | None = None
    candidate_prediction: ArrayLike | None = None
    switch_approved: ArrayLike | None = None
    gate_score: ArrayLike | None = None
    gate_reason: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
