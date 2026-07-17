#!/usr/bin/env python3
"""Run v233 detection and v234 correction without input truth labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.dual_line import (
    BackboneAdapterConfig,
    build_backbone_adapter,
    preprocess_roi_image,
)
from tools.eval_dino_guided_reobserve_v227 import EXPANDED_VIEW_NAMES, _view_boxes
from tools.train_eval_dino_reobserve_promotion_v228 import _device, _layout
from tools.train_eval_error_risk_ranker_v233 import _fused_score
from tools.train_eval_multi_expert_correction_v234 import (
    _combined_score,
    _predict_expert,
)
from tools.train_eval_trajectory_guided_defer_v231 import (
    _base_features,
    _candidate_features,
    _knn_features,
    _load_final_projector,
    _load_trajectory,
    _normality_transform,
    _profile,
    _view_group_features,
)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


@torch.no_grad()
def _predict_packet(
    model: torch.nn.Module,
    cache: dict[str, np.ndarray],
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    model.eval()
    records: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "index",
            "fine_pred",
            "parent_pred",
            "fine_prob",
            "parent_prob",
            "fine_pred_attention",
            "parent_pred_attention",
        )
    }
    for start in range(0, len(indices), batch_size):
        source = indices[start : start + batch_size]
        cls_token = torch.as_tensor(
            cache["cls_token"][source], device=device, dtype=torch.float32
        )
        tile_tokens = torch.as_tensor(
            cache["tile_tokens"][source], device=device, dtype=torch.float32
        )
        output = model(cls_token, tile_tokens)
        fine_prob = F.softmax(output.fine_logits, dim=-1)
        parent_prob = F.softmax(output.parent_logits, dim=-1)
        fine_pred = fine_prob.argmax(dim=-1)
        parent_pred = parent_prob.argmax(dim=-1)
        row = torch.arange(len(source), device=device)
        records["index"].append(source.astype(np.int64))
        records["fine_pred"].append(fine_pred.cpu().numpy())
        records["parent_pred"].append(parent_pred.cpu().numpy())
        records["fine_prob"].append(fine_prob.cpu().numpy())
        records["parent_prob"].append(parent_prob.cpu().numpy())
        records["fine_pred_attention"].append(
            output.fine_spatial_attention[row, fine_pred].cpu().numpy()
        )
        records["parent_pred_attention"].append(
            output.parent_spatial_attention[row, parent_pred].cpu().numpy()
        )
    return {key: np.concatenate(value) for key, value in records.items()}


def _make_unlabeled_views(
    cache: dict[str, np.ndarray],
    base_packet: dict[str, np.ndarray],
    sources: np.ndarray,
    adapter: Any,
    *,
    input_size: int,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    tensors: list[torch.Tensor] = []
    source_rows: list[int] = []
    views: list[str] = []
    bboxes: list[tuple[float, float, float, float]] = []
    for source in sources.tolist():
        boxes = _view_boxes(
            base_packet["parent_pred_attention"][source],
            base_packet["fine_pred_attention"][source],
            policy="expanded",
        )
        seen: set[tuple[float, float, float, float]] = set()
        with Image.open(str(cache["image_path"][source])) as image:
            image = image.convert("RGB")
            for view in EXPANDED_VIEW_NAMES:
                bbox = boxes[view]
                key = tuple(round(value, 5) for value in bbox)
                if key in seen:
                    continue
                seen.add(key)
                tensors.append(preprocess_roi_image(image, bbox, input_size))
                source_rows.append(source)
                views.append(view)
                bboxes.append(bbox)
    cls_parts: list[np.ndarray] = []
    tile_parts: list[np.ndarray] = []
    for start in range(0, len(tensors), batch_size):
        spatial = adapter.encode_spatial(
            torch.stack(tensors[start : start + batch_size]), observer_grid=4
        )
        cls_parts.append(spatial.cls_token.cpu().numpy().astype(np.float16))
        tile_parts.append(spatial.tile_tokens.cpu().numpy().astype(np.float16))
    source_array = np.asarray(source_rows, dtype=np.int64)
    view_cache = {
        "name": np.asarray(
            [f"{cache['name'][source]}::{view}" for source, view in zip(source_rows, views)],
            dtype=str,
        ),
        "cls_token": np.concatenate(cls_parts),
        "tile_tokens": np.concatenate(tile_parts),
    }
    metadata = {
        "source_index": source_array,
        "view": np.asarray(views, dtype=str),
        "bbox": np.asarray(bboxes, dtype=np.float32),
    }
    return view_cache, metadata


def _candidate_feature_rows(
    base_packet: dict[str, np.ndarray],
    candidate_packet: dict[str, np.ndarray],
    metadata: dict[str, np.ndarray],
    features: np.ndarray,
) -> dict[str, np.ndarray]:
    source = metadata["source_index"].astype(np.int64)
    return {
        "features": features,
        "source_index": source,
        "view": metadata["view"],
        "base_parent_pred": base_packet["parent_pred"][source].astype(np.int64),
        "base_fine_pred": base_packet["fine_pred"][source].astype(np.int64),
        "candidate_parent_pred": candidate_packet["parent_pred"].astype(np.int64),
        "candidate_fine_pred": candidate_packet["fine_pred"].astype(np.int64),
    }


def _best_unlabeled_candidate(
    rows: dict[str, np.ndarray],
    candidate_score: np.ndarray,
    base_score: np.ndarray,
    *,
    max_weight: float,
    support_bonus: float,
) -> dict[str, np.ndarray]:
    chosen: list[int] = []
    aggregate: list[float] = []
    for source in np.unique(rows["source_index"]):
        candidates = np.flatnonzero(rows["source_index"] == source)
        pairs = np.column_stack(
            (
                rows["candidate_parent_pred"][candidates],
                rows["candidate_fine_pred"][candidates],
            )
        )
        winner: tuple[float, int, int] | None = None
        for pair in np.unique(pairs, axis=0):
            member = candidates[np.all(pairs == pair, axis=1)]
            values = candidate_score[member]
            pair_score = (
                max_weight * float(values.max())
                + (1.0 - max_weight) * float(values.mean())
                + support_bonus * len(member) / len(candidates)
            )
            candidate = (pair_score, len(member), int(member[np.argmax(values)]))
            if winner is None or candidate > winner:
                winner = candidate
        assert winner is not None
        aggregate.append(winner[0])
        chosen.append(winner[2])
    row = np.asarray(chosen, dtype=np.int64)
    source = rows["source_index"][row]
    return {
        "row": row,
        "source_index": source,
        "candidate_score": np.asarray(aggregate, dtype=np.float32),
        "base_score": base_score[source],
        "base_parent": rows["base_parent_pred"][row],
        "base_fine": rows["base_fine_pred"][row],
        "candidate_parent": rows["candidate_parent_pred"][row],
        "candidate_fine": rows["candidate_fine_pred"][row],
        "view": rows["view"][row],
    }


def run(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    test_cache = _load_npz(args.unlabeled_cache)
    forbidden = sorted(set(test_cache) & {"label", "fine_label", "parent_label", "truth"})
    if forbidden:
        raise ValueError(f"input is not label-free: {forbidden}")
    train_cache = _load_npz(args.train_cache)
    device = _device(args.device)
    checkpoint = torch.load(args.projector_checkpoint, map_location="cpu", weights_only=False)
    layout = _layout(checkpoint)
    mapping = np.asarray(layout.fine_to_parent_index, dtype=np.int64)
    model = _load_final_projector(checkpoint, device)
    indices = np.arange(len(test_cache["name"]), dtype=np.int64)
    packet = _predict_packet(model, test_cache, indices, device, args.eval_batch_size)

    node_features, _, centroids = _load_trajectory(args.trajectory_npz, args.node_csv)
    profile, _, _ = _profile(test_cache["tile_tokens"], packet, node_features, centroids)
    normality = joblib.load(args.normality_model)
    base_reduced, base_normality_risk = _normality_transform(profile, normality)
    base_knn = _knn_features(
        test_cache["cls_token"],
        packet["fine_pred"],
        packet["parent_pred"],
        train_cache["cls_token"],
        train_cache["label"],
        mapping,
        source_index=None,
        k=args.knn,
    )
    base_features = _base_features(
        base_reduced, base_normality_risk, packet, base_knn
    )
    risk_model = joblib.load(args.risk_model)
    direct = risk_model["direct_model"].predict_proba(base_features)[:, 1]
    parent_risk = risk_model["parent_model"].predict_proba(base_features)[:, 1]
    fine_risk = risk_model["fine_model"].predict_proba(base_features)[:, 1]
    risk_score = _fused_score(
        direct,
        parent_risk,
        fine_risk,
        base_normality_risk,
        risk_model["fusion"],
        references=risk_model["score_references"],
    )
    review_threshold = float(
        risk_model["deployment_thresholds"][f"train_budget_{args.review_budget_pct}pct"]
    )
    review = risk_score >= review_threshold
    sources = np.flatnonzero(review)

    correction = joblib.load(args.correction_model)
    base_expert = _predict_expert(correction["base_models"], base_features)
    base_score = _combined_score(base_expert, correction["joint_weight"])
    final_parent = packet["parent_pred"].copy()
    final_fine = packet["fine_pred"].copy()
    switched = np.zeros(len(indices), dtype=bool)
    chosen_view = np.full(len(indices), "keep", dtype="U64")

    if len(sources):
        adapter = build_backbone_adapter(
            BackboneAdapterConfig(
                backbone=args.backbone,
                weights="default",
                input_size=args.input_size,
                device=args.device,
                preprocess_id=f"imagenet_rgb_{args.input_size}",
            )
        )
        view_cache, metadata = _make_unlabeled_views(
            test_cache,
            packet,
            sources,
            adapter,
            input_size=args.input_size,
            batch_size=args.backbone_batch_size,
        )
        candidate_packet = _predict_packet(
            model,
            view_cache,
            np.arange(len(view_cache["name"]), dtype=np.int64),
            device,
            args.eval_batch_size,
        )
        candidate_profile, _, _ = _profile(
            view_cache["tile_tokens"], candidate_packet, node_features, centroids
        )
        candidate_reduced, candidate_risk = _normality_transform(
            candidate_profile, normality
        )
        candidate_knn = _knn_features(
            view_cache["cls_token"],
            candidate_packet["fine_pred"],
            candidate_packet["parent_pred"],
            train_cache["cls_token"],
            train_cache["label"],
            mapping,
            source_index=None,
            k=args.knn,
        )
        group = _view_group_features(candidate_packet, metadata["source_index"])
        candidate_features = _candidate_features(
            base_packet=packet,
            candidate_packet=candidate_packet,
            metadata=metadata,
            base_reduced=base_reduced,
            candidate_reduced=candidate_reduced,
            base_risk=base_normality_risk,
            candidate_risk=candidate_risk,
            base_knn=base_knn,
            candidate_knn=candidate_knn,
            group=group,
            layout=layout,
        )
        rows = _candidate_feature_rows(
            packet, candidate_packet, metadata, candidate_features
        )
        candidate_expert = _predict_expert(
            correction["candidate_models"], candidate_features
        )
        candidate_score = _combined_score(
            candidate_expert, correction["joint_weight"]
        )
        best = _best_unlabeled_candidate(
            rows,
            candidate_score,
            base_score,
            max_weight=correction["pair_consensus_max_weight"],
            support_bonus=correction["pair_consensus_support_bonus"],
        )
        utility = best["candidate_score"] - best["base_score"]
        changed = (best["candidate_parent"] != best["base_parent"]) | (
            best["candidate_fine"] != best["base_fine"]
        )
        approve = changed & (utility >= correction["utility_threshold"])
        source = best["source_index"][approve]
        final_parent[source] = best["candidate_parent"][approve]
        final_fine[source] = best["candidate_fine"][approve]
        switched[source] = True
        chosen_view[best["source_index"]] = best["view"]

    with (args.out_dir / "predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "sample_key",
                "parent_prediction",
                "fine_prediction",
                "parent_index",
                "fine_index",
                "review_requested",
                "switched",
                "chosen_view",
                "risk_score",
            ),
        )
        writer.writeheader()
        for row in indices:
            writer.writerow(
                {
                    "sample_key": str(test_cache["name"][row]),
                    "parent_prediction": layout.parent_classes[int(final_parent[row])],
                    "fine_prediction": layout.fine_classes[int(final_fine[row])],
                    "parent_index": int(final_parent[row]),
                    "fine_index": int(final_fine[row]),
                    "review_requested": bool(review[row]),
                    "switched": bool(switched[row]),
                    "chosen_view": str(chosen_view[row]),
                    "risk_score": float(risk_score[row]),
                }
            )
    summary = {
        "mode": "v235_label_free_inference",
        "input_label_keys_present": False,
        "n": int(len(indices)),
        "review_count": int(review.sum()),
        "switch_count": int(switched.sum()),
        "accuracy_computed": False,
        "output": str(args.out_dir / "predictions.csv"),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unlabeled_cache", type=Path, required=True)
    parser.add_argument("--train_cache", type=Path, required=True)
    parser.add_argument("--normality_model", type=Path, required=True)
    parser.add_argument("--projector_checkpoint", type=Path, required=True)
    parser.add_argument("--trajectory_npz", type=Path, required=True)
    parser.add_argument("--node_csv", type=Path, required=True)
    parser.add_argument("--risk_model", type=Path, required=True)
    parser.add_argument("--correction_model", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--review_budget_pct", type=int, default=10)
    parser.add_argument("--knn", type=int, default=15)
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--backbone_batch_size", type=int, default=48)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
