from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


FEATURE_KEYS = ("rho", "edge_ratio", "int_std")


@dataclass(frozen=True)
class ObserverNeighborhood:
    neighbor_grid: np.ndarray
    boundary_valid: np.ndarray
    offsets: np.ndarray


def tile_id_to_rc(tile_id: int, grid: int) -> tuple[int, int]:
    tile0 = int(tile_id) - 1
    if tile0 < 0 or tile0 >= int(grid) * int(grid):
        raise ValueError(f"tile_id out of range: {tile_id}")
    return tile0 // int(grid), tile0 % int(grid)


def rc_to_tile_id(row: int, col: int, grid: int) -> int:
    return int(row) * int(grid) + int(col) + 1


def build_neighbor_grid(grid: int = 4, *, topology: str = "periodic") -> ObserverNeighborhood:
    """Build a centered 3x3 tile-neighborhood table for every tile.

    `periodic` keeps the compact torus-style relationship that is useful for
    observer-perspective comparisons. `boundary_valid` still marks which 3x3
    positions came from inside the original grid before wrapping/clamping.
    """

    grid = int(grid)
    if grid <= 0:
        raise ValueError("grid must be positive")
    topology = str(topology)
    if topology not in {"periodic", "clamp"}:
        raise ValueError("topology must be periodic|clamp")

    offsets = np.asarray(
        [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)],
        dtype=np.int32,
    ).reshape(3, 3, 2)
    neighbors = np.zeros((grid * grid, 3, 3), dtype=np.int32)
    valid = np.zeros((grid * grid, 3, 3), dtype=np.uint8)

    for tile_id in range(1, grid * grid + 1):
        r, c = tile_id_to_rc(tile_id, grid)
        for rr in range(3):
            for cc in range(3):
                dr, dc = offsets[rr, cc]
                nr = r + int(dr)
                nc = c + int(dc)
                inside = 0 <= nr < grid and 0 <= nc < grid
                valid[tile_id - 1, rr, cc] = 1 if inside else 0
                if topology == "periodic":
                    nr %= grid
                    nc %= grid
                else:
                    nr = min(max(nr, 0), grid - 1)
                    nc = min(max(nc, 0), grid - 1)
                neighbors[tile_id - 1, rr, cc] = rc_to_tile_id(nr, nc, grid)

    return ObserverNeighborhood(neighbor_grid=neighbors, boundary_valid=valid, offsets=offsets)


def phi_to_bucket8(phi_deg: np.ndarray | Iterable[float] | float) -> np.ndarray:
    """Map phi degrees to 8 compass buckets, centered every 45 degrees."""

    phi = np.asarray(phi_deg, dtype=np.float32)
    return np.floor((np.remainder(phi, 360.0) + 22.5) / 45.0).astype(np.int32) % 8


def load_tile_wave_tensor(z: np.lib.npyio.NpzFile, feature_keys: Iterable[str] = FEATURE_KEYS) -> tuple[np.ndarray, list[str]]:
    arrays: list[np.ndarray] = []
    names: list[str] = []
    for key in feature_keys:
        if key in z:
            arr = np.asarray(z[key], dtype=np.float32)
            if arr.ndim != 3:
                raise ValueError(f"{key} must have shape [T,G,G], got {arr.shape}")
            arrays.append(arr)
            names.append(str(key))
    if not arrays:
        raise ValueError("no tile wave features found")
    return np.stack(arrays, axis=-1).astype(np.float32), names


def build_observer_patch(tile_wave_tensor: np.ndarray, neighbor_grid: np.ndarray) -> np.ndarray:
    """Re-index [T,G,G,C] into [T,G*G,3,3,C] centered on each observer tile."""

    x = np.asarray(tile_wave_tensor, dtype=np.float32)
    if x.ndim != 4:
        raise ValueError(f"tile_wave_tensor must be [T,G,G,C], got {x.shape}")
    t, grid_r, grid_c, c = x.shape
    if grid_r != grid_c:
        raise ValueError("tile grid must be square")
    grid = int(grid_r)
    ng = np.asarray(neighbor_grid, dtype=np.int32)
    if ng.shape != (grid * grid, 3, 3):
        raise ValueError(f"neighbor_grid shape mismatch: {ng.shape}")

    flat = x.reshape(t, grid * grid, c)
    out = np.zeros((t, grid * grid, 3, 3, c), dtype=np.float32)
    for tile0 in range(grid * grid):
        idx = ng[tile0].reshape(-1) - 1
        out[:, tile0] = flat[:, idx, :].reshape(t, 3, 3, c)
    return out


def summarize_patch(observer_patch: np.ndarray, boundary_valid: np.ndarray) -> dict[str, np.ndarray]:
    """Small v0.45 diagnostics for observer-neighborhood consistency."""

    patch = np.asarray(observer_patch, dtype=np.float32)
    valid = np.asarray(boundary_valid, dtype=np.float32)[None, :, :, :, None]
    center = patch[:, :, 1:2, 1:2, :]
    diff = np.abs(patch - center)
    valid_sum = np.maximum(valid.sum(axis=(2, 3)), 1.0)
    mean_abs_delta = (diff * valid).sum(axis=(2, 3)) / valid_sum
    max_abs_delta = (diff * valid).max(axis=(2, 3))
    return {
        "neighbor_mean_abs_delta": mean_abs_delta.astype(np.float32),
        "neighbor_max_abs_delta": max_abs_delta.astype(np.float32),
    }
