"""Adapter from legacy observer tile caches to Dual-Line observation samples.

The adapter keeps the legacy observer outputs as the source of observation, but
renames mask/core fields as initial scaffolding so later stages do not treat
them as final detector results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REQUIRED_TILE_KEYS = (
    "phi",
    "rho",
    "edge_ratio",
    "int_std",
    "obj_mask",
    "core_rc",
)


@dataclass(frozen=True)
class ObservationSample:
    """Standard Dual-Line view of one legacy `*.tiles.npz` cache."""

    name: str
    path: Path | None
    phi: np.ndarray
    rho: np.ndarray
    edge_ratio: np.ndarray
    int_std: np.ndarray
    initial_obj_mask: np.ndarray
    initial_core_rc: np.ndarray
    raw_wave_core: np.ndarray
    warnings: tuple[str, ...] = field(default_factory=tuple)
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def T(self) -> int:
        return int(self.phi.shape[0])

    @property
    def grid_size(self) -> int:
        return int(self.initial_obj_mask.shape[0])


def load_observation_sample(
    tiles_npz: str | Path,
    *,
    energy_feature: str = "edge_ratio",
    eps: float = 1e-6,
) -> ObservationSample:
    """Load a legacy `*.tiles.npz` file as an `ObservationSample`."""

    path = Path(tiles_npz)
    with np.load(path, allow_pickle=False) as z:
        sample = observation_sample_from_mapping(
            z,
            name=path.name,
            path=path,
            energy_feature=energy_feature,
            eps=eps,
        )
    return sample


def observation_sample_from_mapping(
    data: Mapping[str, Any],
    *,
    name: str = "<memory>",
    path: str | Path | None = None,
    energy_feature: str = "edge_ratio",
    eps: float = 1e-6,
) -> ObservationSample:
    """Build an observation sample from an npz-like mapping.

    This helper is useful for smoke tests and for future cache builders that
    already hold arrays in memory.
    """

    missing = [key for key in REQUIRED_TILE_KEYS if key not in data]
    if missing:
        raise KeyError(f"missing required tile keys: {missing}")

    warnings: list[str] = []

    phi = _as_float_array(data["phi"], "phi", ndim=1)
    rho = _as_float_array(data["rho"], "rho", ndim=3)
    edge_ratio = _as_float_array(data["edge_ratio"], "edge_ratio", ndim=3)
    int_std = _as_float_array(data["int_std"], "int_std", ndim=3)
    initial_obj_mask = _as_mask(data["obj_mask"])
    initial_core_rc = _as_core_rc(data["core_rc"])

    T = int(phi.shape[0])
    _require_same_shape("rho", rho.shape, (T, rho.shape[1], rho.shape[2]))
    _require_same_shape("edge_ratio", edge_ratio.shape, rho.shape)
    _require_same_shape("int_std", int_std.shape, rho.shape)

    if rho.shape[1] != rho.shape[2]:
        raise ValueError(f"tile maps must be square, got rho shape {rho.shape}")

    G = int(rho.shape[1])
    _require_same_shape("obj_mask", initial_obj_mask.shape, (G, G))

    if G != 4:
        warnings.append(f"unexpected_grid_size:{G}")

    if not np.all(np.isfinite(phi)):
        warnings.append("non_finite_phi")
    for key, arr in (("rho", rho), ("edge_ratio", edge_ratio), ("int_std", int_std)):
        if not np.all(np.isfinite(arr)):
            warnings.append(f"non_finite_{key}")

    if not np.any(initial_obj_mask):
        warnings.append("empty_mask")

    core_valid = _is_valid_core(initial_core_rc, G)
    if not core_valid:
        warnings.append("invalid_core")
    elif not bool(initial_obj_mask[int(initial_core_rc[0]), int(initial_core_rc[1])]):
        warnings.append("core_not_in_mask")

    raw_wave_core = build_raw_wave_core(
        edge_ratio=edge_ratio,
        rho=rho,
        initial_obj_mask=initial_obj_mask,
        energy_feature=edge_ratio if energy_feature == "edge_ratio" else rho,
        eps=eps,
    )

    meta: dict[str, Any] = {
        "source": "tiles_npz",
        "energy_feature": energy_feature,
        "initial_mask_area": float(initial_obj_mask.mean()),
    }
    if "meta" in data:
        meta["legacy_meta_present"] = True

    return ObservationSample(
        name=str(name),
        path=Path(path) if path is not None else None,
        phi=phi,
        rho=rho,
        edge_ratio=edge_ratio,
        int_std=int_std,
        initial_obj_mask=initial_obj_mask,
        initial_core_rc=initial_core_rc,
        raw_wave_core=raw_wave_core,
        warnings=tuple(warnings),
        meta=meta,
    )


def build_raw_wave_core(
    *,
    edge_ratio: np.ndarray,
    rho: np.ndarray,
    initial_obj_mask: np.ndarray,
    energy_feature: np.ndarray | None = None,
    eps: float = 1e-6,
) -> np.ndarray:
    """Build `[edge(phi), dE_mean(phi), dir_loop(phi)]` with shape `(T, 3)`."""

    edge_ratio = np.asarray(edge_ratio, dtype=np.float32)
    rho = np.asarray(rho, dtype=np.float32)
    mask = np.asarray(initial_obj_mask, dtype=bool)
    energy = edge_ratio if energy_feature is None else np.asarray(energy_feature, dtype=np.float32)

    if edge_ratio.ndim != 3:
        raise ValueError(f"edge_ratio must be (T,G,G), got {edge_ratio.shape}")
    if rho.shape != edge_ratio.shape:
        raise ValueError(f"rho shape {rho.shape} != edge_ratio shape {edge_ratio.shape}")
    if energy.shape != edge_ratio.shape:
        raise ValueError(f"energy feature shape {energy.shape} != edge_ratio shape {edge_ratio.shape}")
    if mask.shape != edge_ratio.shape[1:]:
        raise ValueError(f"mask shape {mask.shape} != tile shape {edge_ratio.shape[1:]}")

    edge = _masked_mean_sequence(edge_ratio, mask, eps=eps)
    dE_mean = _energy_separation_sequence(energy, mask, eps=eps)
    dir_loop = _dir_loop_from_base(dE_mean)

    return np.stack([edge, dE_mean, dir_loop], axis=1).astype(np.float32)


def _masked_mean_sequence(values: np.ndarray, mask: np.ndarray, *, eps: float) -> np.ndarray:
    weights = mask.astype(np.float32)
    denom = float(weights.sum())
    if denom <= eps:
        return np.zeros((values.shape[0],), dtype=np.float32)
    return (values * weights[None, :, :]).sum(axis=(1, 2)) / (denom + eps)


def _energy_separation_sequence(energy: np.ndarray, mask: np.ndarray, *, eps: float) -> np.ndarray:
    obj = mask.astype(np.float32)
    bg = (~mask).astype(np.float32)
    obj_sum = float(obj.sum())
    bg_sum = float(bg.sum())

    if obj_sum <= eps or bg_sum <= eps:
        return np.zeros((energy.shape[0],), dtype=np.float32)

    obj_mean = (energy * obj[None, :, :]).sum(axis=(1, 2)) / (obj_sum + eps)
    bg_mean = (energy * bg[None, :, :]).sum(axis=(1, 2)) / (bg_sum + eps)
    all_mean = energy.mean(axis=(1, 2))
    global_floor = float(np.mean(all_mean)) if all_mean.size else 0.0
    scale = np.maximum(all_mean, global_floor) + eps
    return (np.abs(obj_mean - bg_mean) / scale).astype(np.float32)


def _dir_loop_from_base(base: Sequence[float]) -> np.ndarray:
    base_arr = np.asarray(base, dtype=np.float32)
    if base_arr.ndim != 1:
        raise ValueError(f"base waveform must be 1D, got {base_arr.shape}")
    if base_arr.size == 0:
        return np.zeros((0,), dtype=np.float32)
    prev_vals = np.roll(base_arr, 1)
    next_vals = np.roll(base_arr, -1)
    return np.sign(next_vals - prev_vals).astype(np.float32)


def _as_float_array(value: Any, name: str, *, ndim: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got shape {arr.shape}")
    return arr


def _as_mask(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 2:
        raise ValueError(f"obj_mask must be 2D, got shape {arr.shape}")
    return (arr > 0).astype(bool)


def _as_core_rc(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.int32).reshape(-1)
    if arr.shape[0] != 2:
        raise ValueError(f"core_rc must contain two values, got shape {arr.shape}")
    return arr.astype(np.int32)


def _is_valid_core(core_rc: np.ndarray, grid_size: int) -> bool:
    r, c = int(core_rc[0]), int(core_rc[1])
    return 0 <= r < grid_size and 0 <= c < grid_size


def _require_same_shape(name: str, got: tuple[int, ...], expected: tuple[int, ...]) -> None:
    if tuple(got) != tuple(expected):
        raise ValueError(f"{name} shape {got} != expected {expected}")
