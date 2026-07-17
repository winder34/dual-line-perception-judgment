"""Manifest and schema validation utilities.

The goal of this module is to make feature artifacts explicit before we replace
the older CSV-heavy experiment chain.  It keeps the format simple enough for
legacy tools to adopt gradually.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable


MANIFEST_VERSION = "dual_line_artifact_manifest_v1"


@dataclass(slots=True, frozen=True)
class FeatureField:
    """A named feature column or tensor channel."""

    name: str
    dtype: str = "float32"
    shape: list[int | str] = field(default_factory=list)
    required: bool = True
    description: str = ""

    @classmethod
    def from_any(cls, value: str | dict[str, Any]) -> "FeatureField":
        if isinstance(value, str):
            return cls(name=value)
        return cls(
            name=str(value["name"]),
            dtype=str(value.get("dtype", "float32")),
            shape=list(value.get("shape", [])),
            required=bool(value.get("required", True)),
            description=str(value.get("description", "")),
        )


@dataclass(slots=True)
class ArtifactManifest:
    """Portable description of a modular artifact."""

    artifact_type: str
    sample_keys: list[str] = field(default_factory=list)
    feature_schema: list[FeatureField] = field(default_factory=list)
    schema_version: str = MANIFEST_VERSION
    artifact_path: str = ""
    sample_key_column: str = "sample_key"
    backbone_id: str | None = None
    preprocess_id: str | None = None
    scan_policy_id: str | None = None
    candidate_version: str | None = None
    gate_version: str | None = None
    class_map: dict[str, int] = field(default_factory=dict)
    parent_map: dict[str, str] = field(default_factory=dict)
    source_artifacts: dict[str, str] = field(default_factory=dict)
    creation_command: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def sample_count(self) -> int:
        return len(self.sample_keys)

    @property
    def feature_count(self) -> int:
        return len(self.feature_schema)

    @property
    def feature_names(self) -> list[str]:
        return [field.name for field in self.feature_schema]

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["sample_count"] = self.sample_count
        out["feature_count"] = self.feature_count
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArtifactManifest":
        fields = [FeatureField.from_any(x) for x in data.get("feature_schema", [])]
        return cls(
            schema_version=str(data.get("schema_version", MANIFEST_VERSION)),
            artifact_type=str(data["artifact_type"]),
            artifact_path=str(data.get("artifact_path", "")),
            sample_key_column=str(data.get("sample_key_column", "sample_key")),
            sample_keys=[str(x) for x in data.get("sample_keys", [])],
            feature_schema=fields,
            backbone_id=data.get("backbone_id"),
            preprocess_id=data.get("preprocess_id"),
            scan_policy_id=data.get("scan_policy_id"),
            candidate_version=data.get("candidate_version"),
            gate_version=data.get("gate_version"),
            class_map={str(k): int(v) for k, v in data.get("class_map", {}).items()},
            parent_map={str(k): str(v) for k, v in data.get("parent_map", {}).items()},
            source_artifacts={str(k): str(v) for k, v in data.get("source_artifacts", {}).items()},
            creation_command=str(data.get("creation_command", "")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class SchemaValidationResult:
    ok: bool
    n_samples: int = 0
    reference_samples: int = 0
    missing_features: list[str] = field(default_factory=list)
    extra_features: list[str] = field(default_factory=list)
    sample_order_mismatch_count: int = 0
    sample_order_first_mismatches: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_manifest(path: Path | str) -> ArtifactManifest:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return ArtifactManifest.from_dict(json.load(f))


def save_manifest(manifest: ArtifactManifest, path: Path | str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, ensure_ascii=False, indent=2)


def compare_sample_order(sample_keys: Iterable[str], reference_keys: Iterable[str], *, max_examples: int = 10) -> SchemaValidationResult:
    left = [str(x) for x in sample_keys]
    right = [str(x) for x in reference_keys]
    n = min(len(left), len(right))
    mismatches: list[dict[str, Any]] = []
    mismatch_count = abs(len(left) - len(right))
    for i in range(n):
        if left[i] != right[i]:
            mismatch_count += 1
            if len(mismatches) < max_examples:
                mismatches.append({"index": i, "actual": left[i], "reference": right[i]})
    return SchemaValidationResult(
        ok=mismatch_count == 0,
        n_samples=len(left),
        reference_samples=len(right),
        sample_order_mismatch_count=int(mismatch_count),
        sample_order_first_mismatches=mismatches,
    )


def compare_feature_schema(actual_features: Iterable[str], expected_features: Iterable[str], *, strict_extra: bool = False) -> SchemaValidationResult:
    actual = [str(x) for x in actual_features]
    expected = [str(x) for x in expected_features]
    actual_set = set(actual)
    expected_set = set(expected)
    missing = [x for x in expected if x not in actual_set]
    extra = [x for x in actual if x not in expected_set]
    ok = not missing and (not extra if strict_extra else True)
    return SchemaValidationResult(ok=ok, missing_features=missing, extra_features=extra)
