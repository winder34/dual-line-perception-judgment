#!/usr/bin/env python3
"""Evaluate label-free DINO-guided reobservation for v226b review samples.

Ground-truth labels are used only in the final audit.  View generation uses
Parent/Fine attention and validity probabilities from the frozen v226 stack.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from src.dual_line import BackboneAdapterConfig, FrozenDinoV2BackboneAdapter, preprocess_roi_image
from src.dual_line.representation import ConceptLayout
from tools.train_eval_dino_concept_evidence_v226 import _evaluate, _load
from tools.train_eval_dino_state_gate_v226b import _load_final_projector


COMPACT_VIEW_NAMES = ("parent_focus", "fine_focus", "joint_focus", "wide_context")
EXPANDED_VIEW_NAMES = (
    "parent_tight",
    "parent_context",
    "fine_tight",
    "fine_context",
    "joint_tight",
    "joint_context",
    "parent_disagreement",
    "fine_disagreement",
    "secondary_joint",
    "wide_context",
)


def _neighbors(index: int, grid: int) -> list[int]:
    row, col = divmod(index, grid)
    out: list[int] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = row + dr, col + dc
            if 0 <= rr < grid and 0 <= cc < grid:
                out.append(rr * grid + cc)
    return out


def _connected_top(attention: np.ndarray, count: int, grid: int = 4) -> set[int]:
    attention = np.asarray(attention, dtype=np.float32).reshape(-1)
    selected = {int(attention.argmax())}
    while len(selected) < min(int(count), len(attention)):
        frontier = {neighbor for tile in selected for neighbor in _neighbors(tile, grid)} - selected
        candidates = frontier or (set(range(len(attention))) - selected)
        selected.add(max(candidates, key=lambda index: float(attention[index])))
    return selected


def _tiles_to_bbox(tiles: set[int], *, expand_cells: float, grid: int = 4) -> tuple[float, float, float, float]:
    rows = [tile // grid for tile in tiles]
    cols = [tile % grid for tile in tiles]
    expand = float(expand_cells) / float(grid)
    x0 = max(0.0, min(cols) / grid - expand)
    y0 = max(0.0, min(rows) / grid - expand)
    x1 = min(1.0, (max(cols) + 1) / grid + expand)
    y1 = min(1.0, (max(rows) + 1) / grid + expand)
    return x0, y0, x1, y1


def _view_boxes(
    parent_attention: np.ndarray,
    fine_attention: np.ndarray,
    *,
    policy: str,
) -> dict[str, tuple[float, float, float, float]]:
    parent = _connected_top(parent_attention, 4)
    fine = _connected_top(fine_attention, 4)
    joint_attention = 0.5 * np.asarray(parent_attention) + 0.5 * np.asarray(fine_attention)
    joint = _connected_top(joint_attention, 5)
    if policy == "compact":
        return {
            "parent_focus": _tiles_to_bbox(parent, expand_cells=0.25),
            "fine_focus": _tiles_to_bbox(fine, expand_cells=0.25),
            "joint_focus": _tiles_to_bbox(joint, expand_cells=0.25),
            "wide_context": _tiles_to_bbox(parent | fine, expand_cells=1.0),
        }
    if policy != "expanded":
        raise ValueError(f"unknown view policy: {policy}")

    parent_attention = np.asarray(parent_attention, dtype=np.float32)
    fine_attention = np.asarray(fine_attention, dtype=np.float32)
    parent_disagreement = np.maximum(parent_attention - fine_attention, 0.0)
    fine_disagreement = np.maximum(fine_attention - parent_attention, 0.0)
    if float(parent_disagreement.max()) <= 0.0:
        parent_disagreement = parent_attention
    if float(fine_disagreement.max()) <= 0.0:
        fine_disagreement = fine_attention
    secondary_attention = joint_attention.copy()
    primary = int(secondary_attention.argmax())
    secondary_attention[[primary, *_neighbors(primary, 4)]] = -1.0
    return {
        "parent_tight": _tiles_to_bbox(_connected_top(parent_attention, 3), expand_cells=0.10),
        "parent_context": _tiles_to_bbox(_connected_top(parent_attention, 6), expand_cells=0.50),
        "fine_tight": _tiles_to_bbox(_connected_top(fine_attention, 3), expand_cells=0.10),
        "fine_context": _tiles_to_bbox(_connected_top(fine_attention, 6), expand_cells=0.50),
        "joint_tight": _tiles_to_bbox(_connected_top(joint_attention, 4), expand_cells=0.10),
        "joint_context": _tiles_to_bbox(_connected_top(joint_attention, 8), expand_cells=0.50),
        "parent_disagreement": _tiles_to_bbox(_connected_top(parent_disagreement, 4), expand_cells=0.25),
        "fine_disagreement": _tiles_to_bbox(_connected_top(fine_disagreement, 4), expand_cells=0.25),
        "secondary_joint": _tiles_to_bbox(_connected_top(secondary_attention, 4), expand_cells=0.25),
        "wide_context": _tiles_to_bbox(parent | fine, expand_cells=1.0),
    }


def _state_name(parent_correct: bool, fine_correct: bool) -> str:
    if parent_correct and fine_correct:
        return "both_valid"
    if parent_correct:
        return "parent_only"
    if fine_correct:
        return "fine_only"
    return "neither"


def _accuracy_delta(base_correct: np.ndarray, new_correct: np.ndarray) -> dict[str, Any]:
    fixed = int(np.sum(~base_correct & new_correct))
    broken = int(np.sum(base_correct & ~new_correct))
    return {
        "base_correct": int(base_correct.sum()),
        "reobserve_correct": int(new_correct.sum()),
        "fixed": fixed,
        "broken": broken,
        "net": fixed - broken,
    }


def _counter(values: np.ndarray, names: list[str]) -> dict[str, int]:
    counts = Counter(names[int(value)] for value in values.tolist())
    return {name: int(counts.get(name, 0)) for name in names}


def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if args.device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")

    cache = _load(args.test_cache)
    packet = _load(args.evidence_packet)
    scores = _load(args.validity_scores)
    checkpoint = torch.load(args.projector_checkpoint, map_location="cpu", weights_only=False)
    layout_data = checkpoint["layout"]
    layout = ConceptLayout(
        fine_classes=tuple(str(x) for x in layout_data["fine_classes"]),
        parent_classes=tuple(str(x) for x in layout_data["parent_classes"]),
        fine_to_parent_index=tuple(int(x) for x in layout_data["fine_to_parent_index"]),
    )
    names = [str(x) for x in cache["name"].tolist()]
    if names != [str(x) for x in packet["name"].tolist()] or names != [str(x) for x in scores["name"].tolist()]:
        raise ValueError("cache, evidence packet, and validity scores have different sample order")

    parent_valid = scores["parent_validity_probability"].astype(np.float32)
    fine_valid = scores["fine_validity_probability"].astype(np.float32)
    review = (parent_valid < args.parent_threshold) | (fine_valid < args.fine_threshold)
    review_indices = np.flatnonzero(review)
    if len(review_indices) == 0:
        raise RuntimeError("no review samples at the requested thresholds")

    adapter = FrozenDinoV2BackboneAdapter(
        BackboneAdapterConfig(
            backbone=args.backbone,
            weights="default",
            input_size=args.input_size,
            device=args.device,
            preprocess_id=f"imagenet_rgb_{args.input_size}",
        )
    )
    generated_names: list[str] = []
    generated_paths: list[str] = []
    generated_labels: list[int] = []
    generated_views: list[str] = []
    generated_sources: list[int] = []
    generated_bboxes: list[tuple[float, float, float, float]] = []
    cls_parts: list[np.ndarray] = []
    tile_parts: list[np.ndarray] = []
    pending: list[torch.Tensor] = []
    pending_meta: list[tuple[int, str, tuple[float, float, float, float], str]] = []

    def flush() -> None:
        if not pending:
            return
        spatial = adapter.encode_spatial(torch.stack(pending), observer_grid=4)
        cls_parts.append(spatial.cls_token.detach().cpu().numpy().astype(np.float16))
        tile_parts.append(spatial.tile_tokens.detach().cpu().numpy().astype(np.float16))
        for source, view, bbox, image_path in pending_meta:
            generated_names.append(f"{names[source]}::{view}")
            generated_paths.append(image_path)
            generated_labels.append(int(cache["label"][source]))
            generated_views.append(view)
            generated_sources.append(source)
            generated_bboxes.append(bbox)
        pending.clear()
        pending_meta.clear()

    requested_view_names = COMPACT_VIEW_NAMES if args.view_policy == "compact" else EXPANDED_VIEW_NAMES
    unique_view_counts: list[int] = []
    for source in review_indices.tolist():
        boxes = _view_boxes(
            packet["parent_pred_attention"][source],
            packet["fine_pred_attention"][source],
            policy=args.view_policy,
        )
        image_path = str(cache["image_path"][source])
        seen_boxes: set[tuple[float, float, float, float]] = set()
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            for view in requested_view_names:
                bbox = boxes[view]
                bbox_key = tuple(round(value, 5) for value in bbox)
                if bbox_key in seen_boxes:
                    continue
                seen_boxes.add(bbox_key)
                pending.append(preprocess_roi_image(image, bbox, args.input_size))
                pending_meta.append((source, view, bbox, image_path))
                if len(pending) >= args.batch_size:
                    flush()
        unique_view_counts.append(len(seen_boxes))
    flush()

    generated_cache = {
        "name": np.asarray(generated_names, dtype=str),
        "label": np.asarray(generated_labels, dtype=np.int64),
        "class_names": np.asarray(cache["class_names"], dtype=str),
        "image_path": np.asarray(generated_paths, dtype=str),
        "cls_token": np.concatenate(cls_parts, axis=0),
        "tile_tokens": np.concatenate(tile_parts, axis=0),
    }
    projector = _load_final_projector(args.projector_checkpoint, layout, device)
    _, reobserved = _evaluate(
        projector,
        generated_cache,
        np.arange(len(generated_names), dtype=np.int64),
        layout,
        device,
        args.eval_batch_size,
        counterfactual=True,
    )

    source_array = np.asarray(generated_sources, dtype=np.int64)
    view_array = np.asarray(generated_views, dtype=str)
    fine_mapping = np.asarray(layout.fine_to_parent_index, dtype=np.int64)
    records: list[dict[str, Any]] = []
    ensemble_parent: list[int] = []
    ensemble_fine: list[int] = []
    parent_consensus: list[float] = []
    fine_consensus: list[float] = []
    any_parent_correct: list[bool] = []
    any_fine_correct: list[bool] = []
    any_both_correct: list[bool] = []

    for source in review_indices.tolist():
        rows = np.flatnonzero(source_array == source)
        parent_prob = reobserved["parent_prob"][rows]
        fine_prob = reobserved["fine_prob"][rows]
        parent_pred = reobserved["parent_pred"][rows].astype(np.int64)
        fine_pred = reobserved["fine_pred"][rows].astype(np.int64)
        parent_vote = int(parent_prob.mean(axis=0).argmax())
        fine_vote = int(fine_prob.mean(axis=0).argmax())
        parent_ratio = float(np.mean(parent_pred == Counter(parent_pred.tolist()).most_common(1)[0][0]))
        fine_ratio = float(np.mean(fine_pred == Counter(fine_pred.tolist()).most_common(1)[0][0]))
        true_fine = int(packet["fine_label"][source])
        true_parent = int(packet["parent_label"][source])
        base_fine = int(packet["fine_pred"][source])
        base_parent = int(packet["parent_pred"][source])
        ensemble_parent.append(parent_vote)
        ensemble_fine.append(fine_vote)
        parent_consensus.append(parent_ratio)
        fine_consensus.append(fine_ratio)
        any_parent_correct.append(bool(np.any(parent_pred == true_parent)))
        any_fine_correct.append(bool(np.any(fine_pred == true_fine)))
        any_both_correct.append(bool(np.any((parent_pred == true_parent) & (fine_pred == true_fine))))
        records.append(
            {
                "name": names[source],
                "source_index": source,
                "base_state": _state_name(base_parent == true_parent, base_fine == true_fine),
                "parent_validity": float(parent_valid[source]),
                "fine_validity": float(fine_valid[source]),
                "base_parent": layout.parent_classes[base_parent],
                "base_fine": layout.fine_classes[base_fine],
                "reobserve_parent": layout.parent_classes[parent_vote],
                "reobserve_fine": layout.fine_classes[fine_vote],
                "parent_consensus": parent_ratio,
                "fine_consensus": fine_ratio,
                "stable_consensus": bool(parent_ratio >= 0.75 and fine_ratio >= 0.75),
                "parent_changed": bool(parent_vote != base_parent),
                "fine_changed": bool(fine_vote != base_fine),
                "audit_true_parent": layout.parent_classes[true_parent],
                "audit_true_fine": layout.fine_classes[true_fine],
                "audit_parent_correct": bool(parent_vote == true_parent),
                "audit_fine_correct": bool(fine_vote == true_fine),
                "audit_any_parent_correct_view": any_parent_correct[-1],
                "audit_any_fine_correct_view": any_fine_correct[-1],
                "audit_any_both_correct_view": any_both_correct[-1],
            }
        )

    ensemble_parent_array = np.asarray(ensemble_parent, dtype=np.int64)
    ensemble_fine_array = np.asarray(ensemble_fine, dtype=np.int64)
    true_parent = packet["parent_label"][review_indices].astype(np.int64)
    true_fine = packet["fine_label"][review_indices].astype(np.int64)
    base_parent = packet["parent_pred"][review_indices].astype(np.int64)
    base_fine = packet["fine_pred"][review_indices].astype(np.int64)
    base_parent_correct = base_parent == true_parent
    base_fine_correct = base_fine == true_fine
    ensemble_parent_correct = ensemble_parent_array == true_parent
    ensemble_fine_correct = ensemble_fine_array == true_fine
    state_names = np.asarray(
        [_state_name(bool(p), bool(f)) for p, f in zip(base_parent_correct, base_fine_correct)], dtype=str
    )

    by_view: dict[str, Any] = {}
    for view in requested_view_names:
        rows = np.flatnonzero(view_array == view)
        if len(rows) == 0:
            continue
        by_view[view] = {
            "n": int(len(rows)),
            "parent_accuracy_on_review": float(
                np.mean(reobserved["parent_pred"][rows] == reobserved["parent_label"][rows])
            ),
            "fine_accuracy_on_review": float(
                np.mean(reobserved["fine_pred"][rows] == reobserved["fine_label"][rows])
            ),
        }

    by_state: dict[str, Any] = {}
    any_parent_correct_array = np.asarray(any_parent_correct, dtype=bool)
    any_fine_correct_array = np.asarray(any_fine_correct, dtype=bool)
    any_both_correct_array = np.asarray(any_both_correct, dtype=bool)
    for state in ("both_valid", "parent_only", "fine_only", "neither"):
        mask = state_names == state
        by_state[state] = {
            "n": int(mask.sum()),
            "ensemble_parent_correct": int(np.sum(mask & ensemble_parent_correct)),
            "ensemble_fine_correct": int(np.sum(mask & ensemble_fine_correct)),
            "ensemble_both_correct": int(np.sum(mask & ensemble_parent_correct & ensemble_fine_correct)),
            "oracle_parent_correct": int(np.sum(mask & any_parent_correct_array)),
            "oracle_fine_correct": int(np.sum(mask & any_fine_correct_array)),
            "oracle_both_correct": int(np.sum(mask & any_both_correct_array)),
        }

    full_base_parent_correct = packet["parent_pred"].astype(np.int64) == packet["parent_label"].astype(np.int64)
    full_base_fine_correct = packet["fine_pred"].astype(np.int64) == packet["fine_label"].astype(np.int64)
    hypothetical_parent_correct = int(full_base_parent_correct.sum()) + int(
        ensemble_parent_correct.sum() - base_parent_correct.sum()
    )
    hypothetical_fine_correct = int(full_base_fine_correct.sum()) + int(
        ensemble_fine_correct.sum() - base_fine_correct.sum()
    )

    summary = {
        "mode": (
            "v227b_dino_guided_reobserve_expanded_audit"
            if args.view_policy == "expanded"
            else "v227a_dino_guided_reobserve_audit"
        ),
        "gate_or_projector_retrained": False,
        "prediction_switch_enabled": False,
        "view_generation_uses_ground_truth": False,
        "ground_truth_used_for_final_audit_only": True,
        "review_thresholds": {
            "parent": float(args.parent_threshold),
            "fine": float(args.fine_threshold),
        },
        "n_total": int(len(names)),
        "n_review": int(len(review_indices)),
        "review_rate": float(len(review_indices) / len(names)),
        "review_state_counts": dict(Counter(state_names.tolist())),
        "view_policy": args.view_policy,
        "requested_views_per_sample": len(requested_view_names),
        "generated_unique_views_per_sample": {
            "min": int(np.min(unique_view_counts)),
            "mean": float(np.mean(unique_view_counts)),
            "max": int(np.max(unique_view_counts)),
        },
        "view_metrics": by_view,
        "by_base_state": by_state,
        "ensemble_parent": _accuracy_delta(base_parent_correct, ensemble_parent_correct),
        "ensemble_fine": _accuracy_delta(base_fine_correct, ensemble_fine_correct),
        "hypothetical_full_set_if_review_ensemble_applied": {
            "note": "audit only; prediction switching remains disabled",
            "parent_base_accuracy": float(full_base_parent_correct.mean()),
            "parent_reobserve_accuracy": float(hypothetical_parent_correct / len(names)),
            "fine_base_accuracy": float(full_base_fine_correct.mean()),
            "fine_reobserve_accuracy": float(hypothetical_fine_correct / len(names)),
        },
        "oracle": {
            "any_parent_correct_view": int(np.sum(any_parent_correct)),
            "any_fine_correct_view": int(np.sum(any_fine_correct)),
            "any_both_correct_view": int(np.sum(any_both_correct)),
        },
        "stability": {
            "parent_consensus_mean": float(np.mean(parent_consensus)),
            "fine_consensus_mean": float(np.mean(fine_consensus)),
            "both_consensus_ge_0_75": int(
                np.sum((np.asarray(parent_consensus) >= 0.75) & (np.asarray(fine_consensus) >= 0.75))
            ),
            "normal_review_preserved_both": int(
                np.sum((state_names == "both_valid") & ensemble_parent_correct & ensemble_fine_correct)
            ),
            "normal_review_count": int(np.sum(state_names == "both_valid")),
        },
        "ensemble_parent_predictions": _counter(ensemble_parent_array, list(layout.parent_classes)),
        "ensemble_fine_predictions": _counter(ensemble_fine_array, list(layout.fine_classes)),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.out_dir / "reobserve_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    np.savez_compressed(
        args.out_dir / "reobserve_views.npz",
        source_index=source_array,
        source_name=np.asarray([names[index] for index in source_array], dtype=str),
        view=view_array,
        bbox=np.asarray(generated_bboxes, dtype=np.float32),
        parent_prob=reobserved["parent_prob"].astype(np.float32),
        fine_prob=reobserved["fine_prob"].astype(np.float32),
        parent_pred=reobserved["parent_pred"].astype(np.int64),
        fine_pred=reobserved["fine_pred"].astype(np.int64),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_cache", type=Path, required=True)
    parser.add_argument("--evidence_packet", type=Path, required=True)
    parser.add_argument("--validity_scores", type=Path, required=True)
    parser.add_argument("--projector_checkpoint", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--backbone", default="dinov2_vits14", choices=("dinov2_vits14", "dinov2_vitb14"))
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--parent_threshold", type=float, default=0.5)
    parser.add_argument("--fine_threshold", type=float, default=0.5)
    parser.add_argument("--view_policy", choices=("compact", "expanded"), default="compact")
    parser.add_argument("--device", default="auto")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
