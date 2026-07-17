"""Deterministic ROI bootstrap policy for Dual-Line v0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .observer_adapter import ObservationSample


@dataclass(frozen=True)
class BootstrapROI:
    bbox_norm_xyxy: tuple[float, float, float, float]
    bbox_pixel_xyxy: tuple[int, int, int, int] | None
    roi_policy_mode: str
    roi_role: str
    roi_status: str
    fallback_used: bool
    roi_quality: float
    roi_refine_source: str
    reasons: tuple[str, ...]
    meta: dict[str, Any]


def propose_bootstrap_roi(
    sample: ObservationSample,
    *,
    image_size: tuple[int, int] | None = None,
    padding_ratio: float = 0.15,
    square_expand: bool = True,
    fallback_core_tiles: float = 2.0,
    min_bbox_area_ratio: float = 0.03,
    min_aspect_ratio: float = 0.25,
    max_aspect_ratio: float = 4.0,
    eps: float = 1e-6,
) -> BootstrapROI:
    """Create the v0 bootstrap ROI from initial mask/core scaffolding.

    `image_size` is `(width, height)`. When it is omitted, only normalized
    coordinates are returned.
    """

    mask = np.asarray(sample.initial_obj_mask, dtype=bool)
    G = int(mask.shape[0])
    core_r, core_c = int(sample.initial_core_rc[0]), int(sample.initial_core_rc[1])
    reasons: list[str] = []

    if not np.any(mask):
        reasons.append("empty_mask")
        bbox = _core_bbox_norm(core_r, core_c, G, fallback_core_tiles)
        status = "fallback_core"
        fallback = True
    else:
        bbox = _mask_bbox_norm(mask)
        status = "ok"
        fallback = False

    bbox = _pad_bbox(bbox, padding_ratio)
    if square_expand:
        bbox = _square_expand(bbox)
    bbox = _clip_bbox(bbox)

    x0, y0, x1, y1 = bbox
    width = max(0.0, x1 - x0)
    height = max(0.0, y1 - y0)
    area = width * height
    aspect = width / (height + eps)

    if area < min_bbox_area_ratio and "bbox_area_too_small" not in reasons:
        reasons.append("bbox_area_too_small")
    if aspect > max_aspect_ratio and "aspect_too_wide" not in reasons:
        reasons.append("aspect_too_wide")
    if aspect < min_aspect_ratio and "aspect_too_tall" not in reasons:
        reasons.append("aspect_too_tall")

    if not _is_valid_core(core_r, core_c, G):
        reasons.append("invalid_core")
    elif not bool(mask[core_r, core_c]):
        reasons.append("core_not_in_mask")

    fallback_reasons = {"empty_mask", "bbox_area_too_small", "aspect_too_wide", "aspect_too_tall", "invalid_core"}
    if any(reason in fallback_reasons for reason in reasons):
        fallback = True
        status = "fallback_core"
        bbox = _clip_bbox(_square_expand(_pad_bbox(_core_bbox_norm(core_r, core_c, G, fallback_core_tiles), padding_ratio)))

    roi_quality = _roi_quality(mask, bbox, eps=eps)
    bbox_pixel = _to_pixel_bbox(bbox, image_size) if image_size is not None else None

    return BootstrapROI(
        bbox_norm_xyxy=tuple(float(v) for v in bbox),
        bbox_pixel_xyxy=bbox_pixel,
        roi_policy_mode="bootstrap_v0",
        roi_role="bootstrap_seed",
        roi_status=status,
        fallback_used=bool(fallback),
        roi_quality=float(roi_quality),
        roi_refine_source="none",
        reasons=tuple(reasons),
        meta={
            "padding_ratio": float(padding_ratio),
            "square_expand": bool(square_expand),
            "fallback_core_tiles": float(fallback_core_tiles),
            "initial_mask_area": float(mask.mean()),
        },
    )


def _mask_bbox_norm(mask: np.ndarray) -> tuple[float, float, float, float]:
    G = int(mask.shape[0])
    active = np.argwhere(mask)
    r_min = int(active[:, 0].min())
    r_max = int(active[:, 0].max())
    c_min = int(active[:, 1].min())
    c_max = int(active[:, 1].max())
    return (c_min / G, r_min / G, (c_max + 1) / G, (r_max + 1) / G)


def _core_bbox_norm(core_r: int, core_c: int, grid_size: int, tile_span: float) -> tuple[float, float, float, float]:
    if not _is_valid_core(core_r, core_c, grid_size):
        core_r = grid_size // 2
        core_c = grid_size // 2
    center_x = (core_c + 0.5) / float(grid_size)
    center_y = (core_r + 0.5) / float(grid_size)
    side = max(1.0 / grid_size, float(tile_span) / float(grid_size))
    half = side / 2.0
    return (center_x - half, center_y - half, center_x + half, center_y + half)


def _pad_bbox(bbox: tuple[float, float, float, float], padding_ratio: float) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox
    w = x1 - x0
    h = y1 - y0
    px = w * float(padding_ratio)
    py = h * float(padding_ratio)
    return (x0 - px, y0 - py, x1 + px, y1 + py)


def _square_expand(bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    side = max(x1 - x0, y1 - y0)
    half = side / 2.0
    return (cx - half, cy - half, cx + half, cy + half)


def _clip_bbox(bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox
    x0 = min(max(x0, 0.0), 1.0)
    y0 = min(max(y0, 0.0), 1.0)
    x1 = min(max(x1, 0.0), 1.0)
    y1 = min(max(y1, 0.0), 1.0)
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return (x0, y0, x1, y1)


def _to_pixel_bbox(
    bbox: tuple[float, float, float, float],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    width, height = int(image_size[0]), int(image_size[1])
    x0, y0, x1, y1 = bbox
    return (
        int(round(x0 * width)),
        int(round(y0 * height)),
        int(round(x1 * width)),
        int(round(y1 * height)),
    )


def _roi_quality(mask: np.ndarray, bbox: tuple[float, float, float, float], *, eps: float) -> float:
    if not np.any(mask):
        return 0.0
    x0, y0, x1, y1 = bbox
    area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    aspect = max(0.0, x1 - x0) / (max(0.0, y1 - y0) + eps)
    compact_score = min(1.0, float(mask.mean()) / (area + eps))
    area_score = float(np.exp(-abs(area - 0.35)))
    aspect_score = float(np.exp(-abs(np.log(aspect + eps))))
    border_penalty = _border_touch_ratio(mask, eps=eps) * 0.35
    return float(np.clip(compact_score * area_score * aspect_score * (1.0 - border_penalty), 0.0, 1.0))


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


def _is_valid_core(core_r: int, core_c: int, grid_size: int) -> bool:
    return 0 <= core_r < grid_size and 0 <= core_c < grid_size
