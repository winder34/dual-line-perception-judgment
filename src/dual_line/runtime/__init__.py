"""Public single-image runtime contract.

Legacy hub and branch runners remain importable from their concrete modules but
are not loaded eagerly by the public demo package.
"""

from .single_image_inference import (
    SingleImageArtifacts,
    SingleImageInferenceEngine,
    SingleImageSettings,
)

__all__ = [
    "SingleImageArtifacts",
    "SingleImageInferenceEngine",
    "SingleImageSettings",
]
