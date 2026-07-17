"""Structure Vector v0 feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .observer_adapter import ObservationSample


@dataclass(frozen=True)
class StructureFeatureResult:
    vector: np.ndarray
    manifest: list[dict[str, Any]]
    quality: dict[str, float]


FEATURE_SCHEMA: tuple[tuple[str, str], ...] = (
    ("bbox_center_r", "mask_geometry_block"),
    ("bbox_center_c", "mask_geometry_block"),
    ("bbox_h", "mask_geometry_block"),
    ("bbox_w", "mask_geometry_block"),
    ("bbox_area_ratio", "mask_geometry_block"),
    ("bbox_aspect_ratio", "mask_geometry_block"),
    ("mask_fill_ratio", "mask_geometry_block"),
    ("mask_bbox_fill", "mask_geometry_block"),
    ("core_r_norm", "core_location_block"),
    ("core_c_norm", "core_location_block"),
    ("core_to_bbox_dr", "core_location_block"),
    ("core_to_bbox_dc", "core_location_block"),
    ("core_inside_mask", "core_location_block"),
    ("edge_mean", "base_waveform_block"),
    ("edge_std", "base_waveform_block"),
    ("edge_max", "base_waveform_block"),
    ("edge_min", "base_waveform_block"),
    ("edge_range", "base_waveform_block"),
    ("edge_peak_to_mean", "base_waveform_block"),
    ("dE_mean_avg", "base_waveform_block"),
    ("dE_mean_std", "base_waveform_block"),
    ("dE_mean_max", "base_waveform_block"),
    ("dE_mean_min", "base_waveform_block"),
    ("dE_mean_range", "base_waveform_block"),
    ("dominant_phi_sin", "base_waveform_block"),
    ("dominant_phi_cos", "base_waveform_block"),
    ("dominant_phi_strength", "base_waveform_block"),
    ("opposite_phi_consistency", "base_waveform_block"),
    ("dir_pos_ratio", "loop_waveform_block"),
    ("dir_neg_ratio", "loop_waveform_block"),
    ("dir_zero_ratio", "loop_waveform_block"),
    ("sign_transition_ratio_raw", "loop_waveform_block"),
    ("sign_transition_count_nonzero_norm", "loop_waveform_block"),
    ("directional_consistency", "loop_waveform_block"),
    ("signed_edge_mean", "loop_waveform_block"),
    ("signed_edge_abs_mean", "loop_waveform_block"),
    ("signed_edge_energy", "loop_waveform_block"),
    ("signed_edge_balance", "loop_waveform_block"),
    ("signed_edge_absdiff_mean", "loop_waveform_block"),
    ("longest_positive_run_norm", "loop_waveform_block"),
    ("longest_negative_run_norm", "loop_waveform_block"),
    ("waveform_stability", "stability_quality_block"),
    ("border_touch_ratio", "stability_quality_block"),
    ("connectivity_score", "stability_quality_block"),
    ("geometry_quality_score", "stability_quality_block"),
    ("topology_quality", "stability_quality_block"),
)


def build_structure_features(sample: ObservationSample, *, eps: float = 1e-6) -> StructureFeatureResult:
    """Build the explicit Structure Vector v0 from one observation sample."""

    mask = np.asarray(sample.initial_obj_mask, dtype=bool)
    G = int(mask.shape[0])
    active = np.argwhere(mask)
    core_r, core_c = int(sample.initial_core_rc[0]), int(sample.initial_core_rc[1])

    geom = _mask_geometry(mask, active, eps=eps)
    core = _core_features(mask, core_r, core_c, geom, eps=eps)
    wave = _waveform_features(sample.phi, sample.raw_wave_core, eps=eps)
    quality = _quality_features(mask, sample.raw_wave_core[:, 0], eps=eps)

    values = {**geom, **core, **wave, **quality}
    vector = np.asarray([values[name] for name, _block in FEATURE_SCHEMA], dtype=np.float32)
    manifest = [
        {
            "index": i,
            "name": name,
            "block": block,
            "dtype": "float32",
            "source": "observer_adapter.raw_wave_core" if "waveform" in block else "initial_obj_mask/core_rc",
        }
        for i, (name, block) in enumerate(FEATURE_SCHEMA)
    ]

    quality_out = {
        "waveform_stability": float(values["waveform_stability"]),
        "directional_consistency": float(values["directional_consistency"]),
        "geometry_quality_score": float(values["geometry_quality_score"]),
        "topology_quality": float(values["topology_quality"]),
        "connectivity_score": float(values["connectivity_score"]),
        "border_touch_ratio": float(values["border_touch_ratio"]),
    }

    return StructureFeatureResult(vector=vector, manifest=manifest, quality=quality_out)


def feature_manifest() -> list[dict[str, Any]]:
    return [
        {"index": i, "name": name, "block": block, "dtype": "float32"}
        for i, (name, block) in enumerate(FEATURE_SCHEMA)
    ]


def _mask_geometry(mask: np.ndarray, active: np.ndarray, *, eps: float) -> dict[str, float]:
    G = int(mask.shape[0])
    active_count = int(active.shape[0])
    if active_count == 0:
        return {
            "bbox_center_r": 0.5,
            "bbox_center_c": 0.5,
            "bbox_h": 0.0,
            "bbox_w": 0.0,
            "bbox_area_ratio": 0.0,
            "bbox_aspect_ratio": 0.0,
            "mask_fill_ratio": 0.0,
            "mask_bbox_fill": 0.0,
            "_bbox_r_min": 0.0,
            "_bbox_r_max": 0.0,
            "_bbox_c_min": 0.0,
            "_bbox_c_max": 0.0,
        }

    r_min = int(active[:, 0].min())
    r_max = int(active[:, 0].max())
    c_min = int(active[:, 1].min())
    c_max = int(active[:, 1].max())
    bbox_h_tiles = r_max - r_min + 1
    bbox_w_tiles = c_max - c_min + 1
    bbox_tile_count = bbox_h_tiles * bbox_w_tiles
    bbox_h = bbox_h_tiles / float(G)
    bbox_w = bbox_w_tiles / float(G)

    return {
        "bbox_center_r": ((r_min + r_max) / 2.0) / max(G - 1, 1),
        "bbox_center_c": ((c_min + c_max) / 2.0) / max(G - 1, 1),
        "bbox_h": bbox_h,
        "bbox_w": bbox_w,
        "bbox_area_ratio": bbox_h * bbox_w,
        "bbox_aspect_ratio": bbox_w / (bbox_h + eps),
        "mask_fill_ratio": active_count / float(G * G),
        "mask_bbox_fill": active_count / (float(bbox_tile_count) + eps),
        "_bbox_r_min": float(r_min),
        "_bbox_r_max": float(r_max),
        "_bbox_c_min": float(c_min),
        "_bbox_c_max": float(c_max),
    }


def _core_features(
    mask: np.ndarray,
    core_r: int,
    core_c: int,
    geom: dict[str, float],
    *,
    eps: float,
) -> dict[str, float]:
    del eps
    G = int(mask.shape[0])
    valid = 0 <= core_r < G and 0 <= core_c < G
    denom = float(max(G - 1, 1))
    core_r_norm = core_r / denom if valid else 0.0
    core_c_norm = core_c / denom if valid else 0.0
    core_inside = 1.0 if valid and bool(mask[core_r, core_c]) else 0.0
    return {
        "core_r_norm": core_r_norm,
        "core_c_norm": core_c_norm,
        "core_to_bbox_dr": core_r_norm - float(geom["bbox_center_r"]),
        "core_to_bbox_dc": core_c_norm - float(geom["bbox_center_c"]),
        "core_inside_mask": core_inside,
    }


def _waveform_features(phi: np.ndarray, wave_core: np.ndarray, *, eps: float) -> dict[str, float]:
    edge = np.asarray(wave_core[:, 0], dtype=np.float32)
    dE = np.asarray(wave_core[:, 1], dtype=np.float32)
    dir_loop = np.asarray(wave_core[:, 2], dtype=np.float32)
    T = int(edge.shape[0])
    if T == 0:
        return {name: 0.0 for name, block in FEATURE_SCHEMA if block in {"base_waveform_block", "loop_waveform_block"}}

    edge_mean = float(np.mean(edge))
    edge_max = float(np.max(edge))
    edge_min = float(np.min(edge))
    dE_max = float(np.max(dE))
    dE_min = float(np.min(dE))

    t_peak = int(np.argmax(edge))
    phi_peak = float(np.asarray(phi, dtype=np.float32)[t_peak]) if len(phi) > t_peak else 0.0
    phi_rad = np.deg2rad(phi_peak)

    transitions_raw = int(np.count_nonzero(dir_loop != np.roll(dir_loop, -1)))
    nonzero = dir_loop[dir_loop != 0]
    transitions_nonzero = int(np.count_nonzero(nonzero[:-1] != nonzero[1:])) if nonzero.size > 1 else 0
    signed_edge = edge * dir_loop

    return {
        "edge_mean": edge_mean,
        "edge_std": float(np.std(edge)),
        "edge_max": edge_max,
        "edge_min": edge_min,
        "edge_range": edge_max - edge_min,
        "edge_peak_to_mean": edge_max / (edge_mean + eps),
        "dE_mean_avg": float(np.mean(dE)),
        "dE_mean_std": float(np.std(dE)),
        "dE_mean_max": dE_max,
        "dE_mean_min": dE_min,
        "dE_mean_range": dE_max - dE_min,
        "dominant_phi_sin": float(np.sin(phi_rad)),
        "dominant_phi_cos": float(np.cos(phi_rad)),
        "dominant_phi_strength": edge_max / (float(np.sum(edge)) + eps),
        "opposite_phi_consistency": _opposite_phi_consistency(phi, edge, edge_mean=edge_mean, eps=eps),
        "dir_pos_ratio": float(np.mean(dir_loop > 0)),
        "dir_neg_ratio": float(np.mean(dir_loop < 0)),
        "dir_zero_ratio": float(np.mean(dir_loop == 0)),
        "sign_transition_ratio_raw": transitions_raw / float(max(T, 1)),
        "sign_transition_count_nonzero_norm": transitions_nonzero / float(max(T, 1)),
        "directional_consistency": 1.0 - transitions_raw / float(max(T, 1)),
        "signed_edge_mean": float(np.mean(signed_edge)),
        "signed_edge_abs_mean": float(np.mean(np.abs(signed_edge))),
        "signed_edge_energy": float(np.sum(np.abs(signed_edge) * dE) / (np.sum(dE) + eps)),
        "signed_edge_balance": float(np.sum(signed_edge) / (np.sum(np.abs(signed_edge)) + eps)),
        "signed_edge_absdiff_mean": float(np.mean(np.abs(signed_edge - np.roll(signed_edge, 1)))),
        "longest_positive_run_norm": _longest_run(dir_loop, 1.0) / float(max(T, 1)),
        "longest_negative_run_norm": _longest_run(dir_loop, -1.0) / float(max(T, 1)),
    }


def _quality_features(mask: np.ndarray, edge: np.ndarray, *, eps: float) -> dict[str, float]:
    active_count = int(mask.sum())
    border_touch_ratio = _border_touch_ratio(mask, eps=eps)
    connectivity = _connectivity_score(mask, eps=eps)
    geom = _mask_geometry(mask, np.argwhere(mask), eps=eps)
    mask_bbox_fill = float(geom["mask_bbox_fill"])
    edge_diff = np.abs(np.asarray(edge, dtype=np.float32) - np.roll(np.asarray(edge, dtype=np.float32), 1))
    waveform_stability = 1.0 / (1.0 + float(np.mean(edge_diff))) if edge_diff.size else 0.0
    geometry_quality = mask_bbox_fill * (1.0 - border_touch_ratio) if active_count > 0 else 0.0
    topology_quality = connectivity * mask_bbox_fill * (1.0 - border_touch_ratio) if active_count > 0 else 0.0
    return {
        "waveform_stability": waveform_stability,
        "border_touch_ratio": border_touch_ratio,
        "connectivity_score": connectivity,
        "geometry_quality_score": geometry_quality,
        "topology_quality": topology_quality,
    }


def _opposite_phi_consistency(phi: np.ndarray, edge: np.ndarray, *, edge_mean: float, eps: float) -> float:
    if edge.size < 2 or phi.size < 2:
        return 0.0
    diffs = np.diff(np.asarray(phi, dtype=np.float32))
    phi_step = float(np.median(np.abs(diffs))) if diffs.size else 0.0
    if phi_step <= eps:
        return 0.0
    offset_float = 180.0 / phi_step
    offset = int(round(offset_float))
    if abs(offset - offset_float) > 1e-3 or offset <= 0 or offset >= edge.size:
        return 0.0
    opposite_diff = np.abs(edge - np.roll(edge, -offset))
    return float(1.0 - np.mean(opposite_diff) / (edge_mean + eps))


def _longest_run(values: np.ndarray, target: float) -> int:
    longest = 0
    current = 0
    for value in values:
        if float(value) == float(target):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def _border_touch_ratio(mask: np.ndarray, *, eps: float) -> float:
    active_count = int(mask.sum())
    if active_count == 0:
        return 0.0
    border = np.zeros_like(mask, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return float(np.logical_and(mask, border).sum() / (active_count + eps))


def _connectivity_score(mask: np.ndarray, *, eps: float) -> float:
    active_count = int(mask.sum())
    if active_count == 0:
        return 0.0

    seen = np.zeros_like(mask, dtype=bool)
    best = 0
    rows, cols = mask.shape
    for r in range(rows):
        for c in range(cols):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            size = 0
            while stack:
                rr, cc = stack.pop()
                size += 1
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < rows and 0 <= nc < cols and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            best = max(best, size)
    return float(best / (active_count + eps))
