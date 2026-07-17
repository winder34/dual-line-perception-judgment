from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .observer_neighborhood import FEATURE_KEYS, build_neighbor_grid, load_tile_wave_tensor


# Image-coordinate convention used by observer_scan:
# 0=east, 90=south, 180=west, 270=north.
DIRECTION_NAMES = ("nw", "n", "ne", "w", "e", "sw", "s", "se")
DIRECTION_OFFSETS = np.asarray(
    [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)],
    dtype=np.int32,
)
DIRECTION_ANGLES = np.asarray([225.0, 270.0, 315.0, 180.0, 0.0, 135.0, 90.0, 45.0], dtype=np.float32)


@dataclass(frozen=True)
class LocalObserverResult:
    directed_relation: np.ndarray
    local_observer_token: np.ndarray
    neighbor_ids: np.ndarray
    boundary_wrap: np.ndarray
    relation_feature_names: list[str]
    token_feature_names: list[str]


def _circular_distance_deg(phi: np.ndarray, target: float) -> np.ndarray:
    return np.abs((np.asarray(phi, dtype=np.float32) - float(target) + 180.0) % 360.0 - 180.0)


def _direction_weights(phi: np.ndarray, target: float, sigma_deg: float) -> np.ndarray:
    distance = _circular_distance_deg(phi, target)
    weight = np.exp(-0.5 * np.square(distance / max(float(sigma_deg), 1e-6))).astype(np.float32)
    return weight / max(float(weight.sum()), 1e-6)


def _weighted_mean(values: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return np.sum(values * weight[:, None], axis=0)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = a - a.mean(axis=0, keepdims=True)
    bb = b - b.mean(axis=0, keepdims=True)
    denom = np.sqrt(np.sum(aa * aa, axis=0) * np.sum(bb * bb, axis=0)) + 1e-6
    return np.sum(aa * bb, axis=0) / denom


def build_local_observer_tokens(
    *,
    phi: np.ndarray,
    tile_wave_tensor: np.ndarray,
    topology: str = "periodic",
    sigma_deg: float = 22.5,
) -> LocalObserverResult:
    """Build directed 3x3 observer relations and one raw token per tile.

    The relation i->j compares tile i observed toward j with tile j observed
    back toward i. Periodic topology preserves the legacy boundary movement,
    while boundary_wrap records that a relation crossed the finite image edge.
    """

    phi = np.asarray(phi, dtype=np.float32)
    wave = np.asarray(tile_wave_tensor, dtype=np.float32)
    if wave.ndim != 4:
        raise ValueError(f"tile_wave_tensor must be [T,G,G,C], got {wave.shape}")
    t, grid, grid2, channels = wave.shape
    if grid != grid2 or phi.shape != (t,):
        raise ValueError(f"incompatible phi/wave shapes: phi={phi.shape}, wave={wave.shape}")

    neighborhood = build_neighbor_grid(grid=grid, topology=topology)
    flat = wave.reshape(t, grid * grid, channels)

    neighbor_ids = np.zeros((grid * grid, 8), dtype=np.int32)
    boundary_wrap = np.zeros((grid * grid, 8), dtype=np.uint8)
    relation_feature_names = [
        "relative_angle_sin",
        "relative_angle_cos",
        "crosses_boundary",
    ]
    for feature_name in FEATURE_KEYS[:channels]:
        relation_feature_names.extend(
            [
                f"{feature_name}_center_toward_neighbor",
                f"{feature_name}_neighbor_toward_center",
                f"{feature_name}_mutual_signed_delta",
                f"{feature_name}_mutual_abs_delta",
                f"{feature_name}_full_phi_corr",
                f"{feature_name}_full_phi_abs_delta",
            ]
        )

    relation = np.zeros(
        (grid * grid, 8, len(relation_feature_names)),
        dtype=np.float32,
    )
    patch_positions = ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2))

    for center in range(grid * grid):
        for direction, ((patch_r, patch_c), angle) in enumerate(
            zip(patch_positions, DIRECTION_ANGLES.tolist())
        ):
            neighbor_id = int(neighborhood.neighbor_grid[center, patch_r, patch_c])
            neighbor = neighbor_id - 1
            wrapped = 1 - int(neighborhood.boundary_valid[center, patch_r, patch_c])
            neighbor_ids[center, direction] = neighbor_id
            boundary_wrap[center, direction] = wrapped

            forward_weight = _direction_weights(phi, angle, sigma_deg)
            reverse_weight = _direction_weights(phi, (angle + 180.0) % 360.0, sigma_deg)
            center_wave = flat[:, center, :]
            neighbor_wave = flat[:, neighbor, :]
            center_forward = _weighted_mean(center_wave, forward_weight)
            neighbor_reverse = _weighted_mean(neighbor_wave, reverse_weight)
            signed_delta = center_forward - neighbor_reverse

            rad = np.deg2rad(angle)
            values: list[float] = [
                float(np.sin(rad)),
                float(np.cos(rad)),
                float(wrapped),
            ]
            corr = _safe_corr(center_wave, neighbor_wave)
            full_abs_delta = np.mean(np.abs(center_wave - neighbor_wave), axis=0)
            for channel in range(channels):
                values.extend(
                    [
                        float(center_forward[channel]),
                        float(neighbor_reverse[channel]),
                        float(signed_delta[channel]),
                        float(abs(signed_delta[channel])),
                        float(corr[channel]),
                        float(full_abs_delta[channel]),
                    ]
                )
            relation[center, direction] = np.asarray(values, dtype=np.float32)

    token_feature_names: list[str] = []
    token_parts: list[np.ndarray] = []
    center_stats = []
    for channel, feature_name in enumerate(FEATURE_KEYS[:channels]):
        values = flat[:, :, channel]
        for stat_name, stat in [
            ("mean", values.mean(axis=0)),
            ("std", values.std(axis=0)),
            ("min", values.min(axis=0)),
            ("max", values.max(axis=0)),
        ]:
            center_stats.append(stat[:, None])
            token_feature_names.append(f"center_{feature_name}_{stat_name}")
    token_parts.append(np.concatenate(center_stats, axis=1).astype(np.float32))

    # Preserve every directed relation. A later learned projection can compress
    # this raw observer token to 64-128 dimensions without losing provenance.
    token_parts.append(relation.reshape(grid * grid, -1))
    for direction_name in DIRECTION_NAMES:
        token_feature_names.extend(
            [f"{direction_name}::{name}" for name in relation_feature_names]
        )
    token = np.concatenate(token_parts, axis=1).astype(np.float32)
    return LocalObserverResult(
        directed_relation=relation,
        local_observer_token=token,
        neighbor_ids=neighbor_ids,
        boundary_wrap=boundary_wrap,
        relation_feature_names=relation_feature_names,
        token_feature_names=token_feature_names,
    )


def load_and_build(
    z: np.lib.npyio.NpzFile,
    *,
    topology: str = "periodic",
    sigma_deg: float = 22.5,
) -> LocalObserverResult:
    wave, _ = load_tile_wave_tensor(z, feature_keys=FEATURE_KEYS)
    return build_local_observer_tokens(
        phi=np.asarray(z["phi"], dtype=np.float32),
        tile_wave_tensor=wave,
        topology=topology,
        sigma_deg=sigma_deg,
    )
