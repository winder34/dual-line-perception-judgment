"""Representation modules used by the public projector and runtime.

Heavy legacy scorers are available from their concrete modules and are not
imported eagerly here.
"""

from .dino_observer_relation import (
    DINO_OBSERVER_GATE_FEATURE_NAMES,
    DinoObserverRelationAdapter,
    DinoObserverRelationConfig,
    DinoObserverRelationOutput,
    finite_neighbor_mask,
    project_wave_affinity,
    wave_affinity_from_relation_batch,
)
from .dino_concept_evidence import (
    ConceptLayout,
    DinoConceptEvidenceOutput,
    DinoConceptEvidenceProjector,
    symmetric_kl,
    true_class_attention_concentration,
)

__all__ = [
    "DINO_OBSERVER_GATE_FEATURE_NAMES",
    "DinoObserverRelationAdapter",
    "DinoObserverRelationConfig",
    "DinoObserverRelationOutput",
    "finite_neighbor_mask",
    "project_wave_affinity",
    "wave_affinity_from_relation_batch",
    "ConceptLayout",
    "DinoConceptEvidenceOutput",
    "DinoConceptEvidenceProjector",
    "symmetric_kl",
    "true_class_attention_concentration",
]
