"""Dual-Line MVP modules."""

from .backbone_adapters import (
    BackboneAdapterConfig,
    DinoSpatialFeatures,
    FrozenDinoV2BackboneAdapter,
    FrozenTorchvisionBackboneAdapter,
    build_backbone_adapter,
)
from .cnn_encoder import CNNEncoderConfig, FrozenCNNEncoder, preprocess_roi_image
from .fusion_model import FusionHeadV0, FusionHeadV02DualTexture, FusionModelConfig, FusionModelV02Config
from .observer_adapter import ObservationSample, load_observation_sample
from .roi_policy import BootstrapROI, propose_bootstrap_roi
from .runtime_types import (
    BBoxCandidateScoreBatch,
    CandidateBatch,
    ConceptEvidenceBatch,
    EvidenceBatch,
    GateOutput,
    RelationBatch,
    RepresentationBatch,
    ScanBatch,
)
from .structure_features import StructureFeatureResult, build_structure_features, feature_manifest

__all__ = [
    "BootstrapROI",
    "BackboneAdapterConfig",
    "BBoxCandidateScoreBatch",
    "CandidateBatch",
    "CNNEncoderConfig",
    "ConceptEvidenceBatch",
    "DinoSpatialFeatures",
    "EvidenceBatch",
    "FusionHeadV0",
    "FusionHeadV02DualTexture",
    "FusionModelConfig",
    "FusionModelV02Config",
    "FrozenCNNEncoder",
    "FrozenDinoV2BackboneAdapter",
    "FrozenTorchvisionBackboneAdapter",
    "GateOutput",
    "ObservationSample",
    "RelationBatch",
    "RepresentationBatch",
    "ScanBatch",
    "StructureFeatureResult",
    "build_structure_features",
    "build_backbone_adapter",
    "feature_manifest",
    "load_observation_sample",
    "propose_bootstrap_roi",
    "preprocess_roi_image",
]
