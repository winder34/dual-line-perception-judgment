#!/usr/bin/env python3
"""Train and evaluate a trajectory-guided multi-expert reobservation gate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import joblib
import numpy as np
import torch

from src.dual_line import BackboneAdapterConfig, build_backbone_adapter
from tools.build_dino_learning_habit_profile_v230 import (
    _node_distribution,
    _node_features,
    _normalize_rows,
    _weighted_profile,
)
from tools.train_eval_dino_concept_evidence_v226 import _evaluate, _load
from tools.train_eval_dino_reobserve_promotion_v228 import (
    _cross_validated_scores,
    _device,
    _features as _v228_features,
    _layout,
    _make_views,
    _prob_stats,
)
from tools.train_eval_dino_state_gate_v226b import _train_projector_fold
from src.dual_line.representation import DinoConceptEvidenceProjector


def _load_trajectory(
    trajectory_path: Path, node_csv: Path
) -> tuple[np.ndarray, list[str], np.ndarray]:
    import pandas as pd

    with np.load(trajectory_path, allow_pickle=False) as loaded:
        trajectory = {key: loaded[key] for key in loaded.files}
    audit = pd.read_csv(node_csv)
    node_features, feature_names, _ = _node_features(trajectory, audit)
    return (
        node_features,
        feature_names,
        _normalize_rows(trajectory["node_centroid"].astype(np.float32)),
    )


def _load_final_projector(
    checkpoint: dict[str, Any], device: torch.device
) -> DinoConceptEvidenceProjector:
    layout = _layout(checkpoint)
    config = checkpoint["config"]
    model = DinoConceptEvidenceProjector(
        embedding_dim=int(config["embedding_dim"]),
        parent_classes=len(layout.parent_classes),
        fine_classes=len(layout.fine_classes),
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["state"])
    model.eval()
    return model


def _profile(
    tile_tokens: np.ndarray,
    packet: dict[str, np.ndarray],
    node_features: np.ndarray,
    centroids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = tile_tokens.shape[:2]
    normalized = _normalize_rows(tile_tokens.reshape(-1, tile_tokens.shape[-1]).astype(np.float32))
    assignment = (normalized @ centroids.T).argmax(axis=1).reshape(shape)
    parent_attention = packet["parent_pred_attention"].astype(np.float32)
    fine_attention = packet["fine_pred_attention"].astype(np.float32)
    parent_dist = _node_distribution(parent_attention, assignment, len(centroids))
    fine_dist = _node_distribution(fine_attention, assignment, len(centroids))
    parent_mean, parent_std, parent_max = _weighted_profile(
        parent_attention, assignment, node_features
    )
    fine_mean, fine_std, fine_max = _weighted_profile(fine_attention, assignment, node_features)
    midpoint = 0.5 * (parent_dist + fine_dist)
    js = 0.5 * (
        np.sum(parent_dist * np.log((parent_dist + 1.0e-8) / (midpoint + 1.0e-8)), axis=1)
        + np.sum(fine_dist * np.log((fine_dist + 1.0e-8) / (midpoint + 1.0e-8)), axis=1)
    )
    l1 = np.abs(parent_dist - fine_dist).sum(axis=1)
    cosine = np.sum(parent_dist * fine_dist, axis=1) / np.maximum(
        np.linalg.norm(parent_dist, axis=1) * np.linalg.norm(fine_dist, axis=1), 1.0e-8
    )
    profile = np.column_stack(
        (
            parent_mean,
            parent_std,
            parent_max,
            fine_mean,
            fine_std,
            fine_max,
            parent_mean - fine_mean,
            np.abs(parent_mean - fine_mean),
            js,
            l1,
            cosine,
            packet["parent_prob"].max(axis=1),
            packet["fine_prob"].max(axis=1),
        )
    ).astype(np.float32)
    return profile, parent_dist, fine_dist


def _normality_transform(
    profile: np.ndarray, normality: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    reduced = normality["pca"].transform(normality["scaler"].transform(profile)).astype(np.float32)
    risk = -normality["normality"].score_samples(reduced)
    return reduced, risk.astype(np.float32)


def _knn_features(
    query: np.ndarray,
    predicted_fine: np.ndarray,
    predicted_parent: np.ndarray,
    train_cls: np.ndarray,
    train_fine: np.ndarray,
    fine_to_parent: np.ndarray,
    *,
    source_index: np.ndarray | None,
    k: int,
) -> np.ndarray:
    query = _normalize_rows(query.astype(np.float32))
    train_cls = _normalize_rows(train_cls.astype(np.float32))
    output: list[np.ndarray] = []
    train_parent = fine_to_parent[train_fine]
    for start in range(0, len(query), 256):
        stop = min(len(query), start + 256)
        similarity = query[start:stop] @ train_cls.T
        if source_index is not None:
            rows = np.arange(stop - start)
            similarity[rows, source_index[start:stop]] = -np.inf
        neighbor = np.argpartition(similarity, -k, axis=1)[:, -k:]
        neighbor_similarity = np.take_along_axis(similarity, neighbor, axis=1)
        order = np.argsort(neighbor_similarity, axis=1)[:, ::-1]
        neighbor = np.take_along_axis(neighbor, order, axis=1)
        neighbor_similarity = np.take_along_axis(neighbor_similarity, order, axis=1)
        fine_support = np.mean(
            train_fine[neighbor] == predicted_fine[start:stop, None], axis=1
        )
        parent_support = np.mean(
            train_parent[neighbor] == predicted_parent[start:stop, None], axis=1
        )
        entropy: list[float] = []
        for labels in train_fine[neighbor]:
            counts = np.bincount(labels, minlength=int(train_fine.max()) + 1).astype(np.float32)
            probability = counts[counts > 0] / max(1.0, counts.sum())
            entropy.append(float(-(probability * np.log(probability)).sum() / np.log(k)))
        output.append(
            np.column_stack(
                (
                    fine_support,
                    parent_support,
                    neighbor_similarity[:, 0],
                    neighbor_similarity.mean(axis=1),
                    np.asarray(entropy, dtype=np.float32),
                )
            ).astype(np.float32)
        )
    return np.concatenate(output)


def _view_group_features(
    packet: dict[str, np.ndarray], source_index: np.ndarray
) -> np.ndarray:
    result = np.zeros((len(source_index), 8), dtype=np.float32)
    parent_pred = packet["parent_pred"].astype(np.int64)
    fine_pred = packet["fine_pred"].astype(np.int64)
    for source in np.unique(source_index):
        rows = np.flatnonzero(source_index == source)
        parent_mean = packet["parent_prob"][rows].mean(axis=0)
        fine_mean = packet["fine_prob"][rows].mean(axis=0)
        parent_entropy = _prob_stats(parent_mean)[2]
        fine_entropy = _prob_stats(fine_mean)[2]
        for row in rows:
            result[row] = (
                np.mean(parent_pred[rows] == parent_pred[row]),
                np.mean(fine_pred[rows] == fine_pred[row]),
                parent_mean[parent_pred[row]],
                fine_mean[fine_pred[row]],
                parent_entropy,
                fine_entropy,
                len(np.unique(parent_pred[rows])) / len(rows),
                len(np.unique(fine_pred[rows])) / len(rows),
            )
    return result


def _base_features(
    reduced: np.ndarray,
    risk: np.ndarray,
    packet: dict[str, np.ndarray],
    knn: np.ndarray,
) -> np.ndarray:
    stats = []
    for parent_probability, fine_probability in zip(packet["parent_prob"], packet["fine_prob"]):
        stats.append((*_prob_stats(parent_probability), *_prob_stats(fine_probability)))
    return np.column_stack((reduced, risk, np.asarray(stats, dtype=np.float32), knn)).astype(np.float32)


def _candidate_features(
    *,
    base_packet: dict[str, np.ndarray],
    candidate_packet: dict[str, np.ndarray],
    metadata: dict[str, np.ndarray],
    base_reduced: np.ndarray,
    candidate_reduced: np.ndarray,
    base_risk: np.ndarray,
    candidate_risk: np.ndarray,
    base_knn: np.ndarray,
    candidate_knn: np.ndarray,
    group: np.ndarray,
    layout: Any,
) -> np.ndarray:
    by_source = {int(index): position for position, index in enumerate(base_packet["index"].tolist())}
    rows: list[np.ndarray] = []
    for row, source in enumerate(metadata["source_index"].tolist()):
        position = by_source[int(source)]
        conventional = _v228_features(
            base_parent_prob=base_packet["parent_prob"][position],
            base_fine_prob=base_packet["fine_prob"][position],
            candidate_parent_prob=candidate_packet["parent_prob"][row],
            candidate_fine_prob=candidate_packet["fine_prob"][row],
            parent_validity=float(np.exp(-base_risk[position])),
            fine_validity=float(np.exp(-base_risk[position])),
            bbox=metadata["bbox"][row],
            view=str(metadata["view"][row]),
            layout=layout,
        )
        rows.append(
            np.concatenate(
                (
                    conventional,
                    base_reduced[position],
                    candidate_reduced[row],
                    candidate_reduced[row] - base_reduced[position],
                    np.asarray(
                        (
                            base_risk[position],
                            candidate_risk[row],
                            base_risk[position] - candidate_risk[row],
                        ),
                        dtype=np.float32,
                    ),
                    base_knn[position],
                    candidate_knn[row],
                    group[row],
                )
            )
        )
    return np.stack(rows).astype(np.float32)


def _candidate_rows(
    *,
    base_packet: dict[str, np.ndarray],
    candidate_packet: dict[str, np.ndarray],
    metadata: dict[str, np.ndarray],
    features: np.ndarray,
    folds: np.ndarray,
) -> dict[str, np.ndarray]:
    by_source = {int(index): position for position, index in enumerate(base_packet["index"].tolist())}
    position = np.asarray([by_source[int(source)] for source in metadata["source_index"]], dtype=np.int64)
    source = metadata["source_index"].astype(np.int64)
    return {
        "features": features,
        "source_index": source,
        "fold": folds[source],
        "view": metadata["view"],
        "base_parent_pred": base_packet["parent_pred"][position].astype(np.int64),
        "base_fine_pred": base_packet["fine_pred"][position].astype(np.int64),
        "candidate_parent_pred": candidate_packet["parent_pred"].astype(np.int64),
        "candidate_fine_pred": candidate_packet["fine_pred"].astype(np.int64),
        "true_parent": base_packet["parent_label"][position].astype(np.int64),
        "true_fine": base_packet["fine_label"][position].astype(np.int64),
        "parent_target": (
            candidate_packet["parent_pred"].astype(np.int64)
            == base_packet["parent_label"][position].astype(np.int64)
        ).astype(np.int64),
        "fine_target": (
            candidate_packet["fine_pred"].astype(np.int64)
            == base_packet["fine_label"][position].astype(np.int64)
        ).astype(np.int64),
    }


def _concat(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


def _best_action(
    rows: dict[str, np.ndarray], scores: np.ndarray, base_score: np.ndarray, branch: str
) -> dict[str, np.ndarray]:
    chosen: list[int] = []
    for source in np.unique(rows["source_index"]):
        candidates = np.flatnonzero(rows["source_index"] == source)
        chosen.append(int(candidates[np.argmax(scores[candidates])]))
    index = np.asarray(chosen, dtype=np.int64)
    source = rows["source_index"][index]
    return {
        "row": index,
        "source_index": source,
        "candidate_score": scores[index],
        "base_score": base_score[source],
        "utility_delta": scores[index] - base_score[source],
        "base_pred": rows[f"base_{branch}_pred"][index],
        "candidate_pred": rows[f"candidate_{branch}_pred"][index],
        "truth": rows[f"true_{branch}"][index],
        "view": rows["view"][index],
    }


def _policy(best: dict[str, np.ndarray], threshold: float) -> dict[str, Any]:
    switch = (best["utility_delta"] >= threshold) & (
        best["candidate_pred"] != best["base_pred"]
    )
    final = np.where(switch, best["candidate_pred"], best["base_pred"])
    base_correct = best["base_pred"] == best["truth"]
    final_correct = final == best["truth"]
    fixed = int(np.sum(~base_correct & final_correct))
    broken = int(np.sum(base_correct & ~final_correct))
    return {
        "threshold": float(threshold),
        "n": int(len(final)),
        "switch_count": int(switch.sum()),
        "fixed": fixed,
        "broken": broken,
        "net": fixed - broken,
    }


def _threshold(
    best: dict[str, np.ndarray], broken_penalty: float, min_utility_delta: float
) -> tuple[float, dict[str, Any]]:
    candidates = np.unique(
        np.concatenate(([min_utility_delta, 2.0], best["utility_delta"][best["utility_delta"] >= min_utility_delta]))
    )
    winner: tuple[tuple[float, int, int, int, float], dict[str, Any]] | None = None
    for threshold in candidates:
        metrics = _policy(best, float(threshold))
        rank = (
            metrics["fixed"] - broken_penalty * metrics["broken"],
            metrics["fixed"],
            -metrics["broken"],
            -metrics["switch_count"],
            float(threshold),
        )
        if winner is None or rank > winner[0]:
            winner = (rank, metrics)
    assert winner is not None
    return float(winner[1]["threshold"]), winner[1]


def run(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_cache = _load(args.train_cache)
    test_cache = _load(args.test_cache)
    oof = _load(args.oof_features)
    folds = oof["fold"].astype(np.int64)
    checkpoint = torch.load(args.projector_checkpoint, map_location="cpu", weights_only=False)
    layout = _layout(checkpoint)
    mapping = np.asarray(layout.fine_to_parent_index, dtype=np.int64)
    normality = joblib.load(args.normality_model)
    node_features, _, centroids = _load_trajectory(args.trajectory_npz, args.node_csv)
    with np.load(args.train_profile, allow_pickle=False) as loaded:
        train_profile = loaded["habit_profile"].astype(np.float32)
    _, train_risk = _normality_transform(train_profile, normality)
    review_risk_threshold = float(np.quantile(train_risk, args.review_quantile))
    adapter = build_backbone_adapter(
        BackboneAdapterConfig(
            backbone=args.backbone,
            weights="default",
            input_size=args.input_size,
            device=args.device,
            preprocess_id=f"imagenet_rgb_{args.input_size}",
        )
    )
    projector_args = SimpleNamespace(
        seed=args.projector_seed,
        projector_hidden_dim=int(checkpoint["config"]["hidden_dim"]),
        projector_dropout=float(checkpoint["config"]["dropout"]),
        projector_lr=args.projector_lr,
        projector_batch_size=args.projector_batch_size,
        projector_epochs=args.projector_epochs,
        tile_drop_rate=args.tile_drop_rate,
        concentration_limit=args.concentration_limit,
    )

    base_parts: list[dict[str, np.ndarray]] = []
    candidate_parts: list[dict[str, np.ndarray]] = []
    train_cls = train_cache["cls_token"].astype(np.float32)
    train_fine = train_cache["label"].astype(np.int64)
    for fold in sorted(np.unique(folds).tolist()):
        heldout = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        model = _train_projector_fold(
            train_cache, layout, training, args=projector_args, fold=int(fold), device=device
        )
        _, base_packet = _evaluate(
            model, train_cache, heldout, layout, device, args.eval_batch_size, counterfactual=False
        )
        base_profile, _, _ = _profile(
            train_cache["tile_tokens"][heldout], base_packet, node_features, centroids
        )
        base_reduced, base_risk = _normality_transform(base_profile, normality)
        base_knn = _knn_features(
            train_cache["cls_token"][heldout],
            base_packet["fine_pred"],
            base_packet["parent_pred"],
            train_cls,
            train_fine,
            mapping,
            source_index=heldout,
            k=args.knn,
        )
        base_feature = _base_features(base_reduced, base_risk, base_packet, base_knn)
        base_parts.append(
            {
                "features": base_feature,
                "source_index": heldout,
                "fold": folds[heldout],
                "parent_target": (base_packet["parent_pred"] == base_packet["parent_label"]).astype(np.int64),
                "fine_target": (base_packet["fine_pred"] == base_packet["fine_label"]).astype(np.int64),
            }
        )
        fold_threshold = float(np.quantile(base_risk, args.review_quantile))
        reviewed_position = np.flatnonzero(base_risk >= fold_threshold)
        sources = heldout[reviewed_position]
        view_cache, metadata = _make_views(
            train_cache,
            base_packet,
            sources,
            adapter,
            input_size=args.input_size,
            batch_size=args.dino_batch_size,
        )
        _, candidate_packet = _evaluate(
            model,
            view_cache,
            np.arange(len(view_cache["label"]), dtype=np.int64),
            layout,
            device,
            args.eval_batch_size,
            counterfactual=False,
        )
        candidate_profile, _, _ = _profile(
            view_cache["tile_tokens"], candidate_packet, node_features, centroids
        )
        candidate_reduced, candidate_risk = _normality_transform(candidate_profile, normality)
        candidate_knn = _knn_features(
            view_cache["cls_token"],
            candidate_packet["fine_pred"],
            candidate_packet["parent_pred"],
            train_cls,
            train_fine,
            mapping,
            source_index=metadata["source_index"],
            k=args.knn,
        )
        base_position = np.asarray(
            [{int(index): position for position, index in enumerate(heldout)}[int(source)] for source in metadata["source_index"]],
            dtype=np.int64,
        )
        group = _view_group_features(candidate_packet, metadata["source_index"])
        features = _candidate_features(
            base_packet=base_packet,
            candidate_packet=candidate_packet,
            metadata=metadata,
            base_reduced=base_reduced,
            candidate_reduced=candidate_reduced,
            base_risk=base_risk,
            candidate_risk=candidate_risk,
            base_knn=base_knn,
            candidate_knn=candidate_knn,
            group=group,
            layout=layout,
        )
        candidate_parts.append(
            _candidate_rows(
                base_packet=base_packet,
                candidate_packet=candidate_packet,
                metadata=metadata,
                features=features,
                folds=folds,
            )
        )
        print(
            f"[OOF] fold={fold} base={len(heldout)} review={len(sources)} candidates={len(features)}",
            flush=True,
        )

    base_rows = _concat(base_parts)
    order = np.argsort(base_rows["source_index"])
    base_rows = {key: value[order] for key, value in base_rows.items()}
    candidate_rows = _concat(candidate_parts)
    np.savez_compressed(args.out_dir / "oof_base_rows_v231.npz", **base_rows)
    np.savez_compressed(args.out_dir / "oof_candidate_rows_raw_v231.npz", **candidate_rows)
    base_parent_score, base_parent_model = _cross_validated_scores(
        base_rows["features"], base_rows["parent_target"], base_rows["fold"], args.seed
    )
    base_fine_score, base_fine_model = _cross_validated_scores(
        base_rows["features"], base_rows["fine_target"], base_rows["fold"], args.seed + 10
    )
    candidate_parent_score, candidate_parent_model = _cross_validated_scores(
        candidate_rows["features"], candidate_rows["parent_target"], candidate_rows["fold"], args.seed + 20
    )
    candidate_fine_score, candidate_fine_model = _cross_validated_scores(
        candidate_rows["features"], candidate_rows["fine_target"], candidate_rows["fold"], args.seed + 30
    )
    parent_best = _best_action(candidate_rows, candidate_parent_score, base_parent_score, "parent")
    fine_best = _best_action(candidate_rows, candidate_fine_score, base_fine_score, "fine")
    parent_threshold, parent_oof = _threshold(
        parent_best, args.broken_penalty, args.min_utility_delta
    )
    fine_threshold, fine_oof = _threshold(
        fine_best, args.broken_penalty, args.min_utility_delta
    )

    final_model = _load_final_projector(checkpoint, device)
    test_indices = np.arange(len(test_cache["label"]), dtype=np.int64)
    _, test_packet = _evaluate(
        final_model, test_cache, test_indices, layout, device, args.eval_batch_size, counterfactual=False
    )
    test_profile, _, _ = _profile(test_cache["tile_tokens"], test_packet, node_features, centroids)
    test_reduced, test_risk = _normality_transform(test_profile, normality)
    test_knn = _knn_features(
        test_cache["cls_token"],
        test_packet["fine_pred"],
        test_packet["parent_pred"],
        train_cls,
        train_fine,
        mapping,
        source_index=None,
        k=args.knn,
    )
    test_base_features = _base_features(test_reduced, test_risk, test_packet, test_knn)
    test_base_parent_score = base_parent_model.predict_proba(test_base_features)[:, 1].astype(np.float32)
    test_base_fine_score = base_fine_model.predict_proba(test_base_features)[:, 1].astype(np.float32)
    test_review = test_risk >= review_risk_threshold
    test_sources = np.flatnonzero(test_review)
    test_view_cache, test_metadata = _make_views(
        test_cache,
        test_packet,
        test_sources,
        adapter,
        input_size=args.input_size,
        batch_size=args.dino_batch_size,
    )
    _, test_candidate_packet = _evaluate(
        final_model,
        test_view_cache,
        np.arange(len(test_view_cache["label"]), dtype=np.int64),
        layout,
        device,
        args.eval_batch_size,
        counterfactual=False,
    )
    test_candidate_profile, _, _ = _profile(
        test_view_cache["tile_tokens"], test_candidate_packet, node_features, centroids
    )
    test_candidate_reduced, test_candidate_risk = _normality_transform(
        test_candidate_profile, normality
    )
    test_candidate_knn = _knn_features(
        test_view_cache["cls_token"],
        test_candidate_packet["fine_pred"],
        test_candidate_packet["parent_pred"],
        train_cls,
        train_fine,
        mapping,
        source_index=None,
        k=args.knn,
    )
    test_group = _view_group_features(test_candidate_packet, test_metadata["source_index"])
    test_candidate_features = _candidate_features(
        base_packet=test_packet,
        candidate_packet=test_candidate_packet,
        metadata=test_metadata,
        base_reduced=test_reduced,
        candidate_reduced=test_candidate_reduced,
        base_risk=test_risk,
        candidate_risk=test_candidate_risk,
        base_knn=test_knn,
        candidate_knn=test_candidate_knn,
        group=test_group,
        layout=layout,
    )
    test_rows = _candidate_rows(
        base_packet=test_packet,
        candidate_packet=test_candidate_packet,
        metadata=test_metadata,
        features=test_candidate_features,
        folds=np.zeros(len(test_cache["label"]), dtype=np.int64),
    )
    test_parent_candidate_score = candidate_parent_model.predict_proba(test_candidate_features)[:, 1]
    test_fine_candidate_score = candidate_fine_model.predict_proba(test_candidate_features)[:, 1]
    test_parent_best = _best_action(
        test_rows, test_parent_candidate_score, test_base_parent_score, "parent"
    )
    test_fine_best = _best_action(
        test_rows, test_fine_candidate_score, test_base_fine_score, "fine"
    )
    test_parent_policy = _policy(test_parent_best, parent_threshold)
    test_fine_policy = _policy(test_fine_best, fine_threshold)

    full_parent = test_packet["parent_pred"].astype(np.int64).copy()
    full_fine = test_packet["fine_pred"].astype(np.int64).copy()
    parent_switch = (test_parent_best["utility_delta"] >= parent_threshold) & (
        test_parent_best["candidate_pred"] != test_parent_best["base_pred"]
    )
    fine_switch = (test_fine_best["utility_delta"] >= fine_threshold) & (
        test_fine_best["candidate_pred"] != test_fine_best["base_pred"]
    )
    full_parent[test_parent_best["source_index"][parent_switch]] = test_parent_best["candidate_pred"][parent_switch]
    full_fine[test_fine_best["source_index"][fine_switch]] = test_fine_best["candidate_pred"][fine_switch]
    true_fine = test_cache["label"].astype(np.int64)
    true_parent = mapping[true_fine]
    base_parent = test_packet["parent_pred"].astype(np.int64)
    base_fine = test_packet["fine_pred"].astype(np.int64)
    base_invalid = ~((base_parent == true_parent) & (base_fine == true_fine))
    summary = {
        "mode": "v231a_trajectory_guided_multi_expert_defer",
        "backbone": args.backbone,
        "test_used_for_training_or_threshold_selection": False,
        "test_weight_update_enabled": False,
        "review_quantile_from_train": args.review_quantile,
        "review_risk_threshold": review_risk_threshold,
        "broken_penalty": args.broken_penalty,
        "min_utility_delta": args.min_utility_delta,
        "train_oof": {
            "base_rows": int(len(base_rows["features"])),
            "review_samples": int(len(np.unique(candidate_rows["source_index"]))),
            "candidate_rows": int(len(candidate_rows["features"])),
            "parent_policy": parent_oof,
            "fine_policy": fine_oof,
        },
        "test_detection": {
            "review_samples": int(test_review.sum()),
            "base_invalid_total": int(base_invalid.sum()),
            "base_invalid_captured": int(np.sum(test_review & base_invalid)),
            "false_review": int(np.sum(test_review & ~base_invalid)),
        },
        "test_policy": {"parent": test_parent_policy, "fine": test_fine_policy},
        "full_test": {
            "base_parent_accuracy": float(np.mean(base_parent == true_parent)),
            "final_parent_accuracy": float(np.mean(full_parent == true_parent)),
            "base_fine_accuracy": float(np.mean(base_fine == true_fine)),
            "final_fine_accuracy": float(np.mean(full_fine == true_fine)),
            "base_both_accuracy": float(np.mean((base_parent == true_parent) & (base_fine == true_fine))),
            "final_both_accuracy": float(np.mean((full_parent == true_parent) & (full_fine == true_fine))),
        },
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    joblib.dump(
        {
            "base_parent_model": base_parent_model,
            "base_fine_model": base_fine_model,
            "candidate_parent_model": candidate_parent_model,
            "candidate_fine_model": candidate_fine_model,
            "parent_threshold": parent_threshold,
            "fine_threshold": fine_threshold,
            "review_risk_threshold": review_risk_threshold,
        },
        args.out_dir / "trajectory_guided_defer_v231.pkl",
    )
    np.savez_compressed(
        args.out_dir / "oof_action_rows_v231.npz",
        **candidate_rows,
        base_parent_oof_score=base_parent_score[candidate_rows["source_index"]],
        base_fine_oof_score=base_fine_score[candidate_rows["source_index"]],
        parent_score=candidate_parent_score,
        fine_score=candidate_fine_score,
    )
    with (args.out_dir / "test_action_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "branch", "source_index", "name", "base_score", "candidate_score",
                "utility_delta", "threshold", "view", "switched", "base", "candidate", "truth",
            ),
        )
        writer.writeheader()
        for branch, best, threshold, switched in (
            ("parent", test_parent_best, parent_threshold, parent_switch),
            ("fine", test_fine_best, fine_threshold, fine_switch),
        ):
            for row, source in enumerate(best["source_index"]):
                writer.writerow(
                    {
                        "branch": branch,
                        "source_index": int(source),
                        "name": str(test_cache["name"][source]),
                        "base_score": float(best["base_score"][row]),
                        "candidate_score": float(best["candidate_score"][row]),
                        "utility_delta": float(best["utility_delta"][row]),
                        "threshold": float(threshold),
                        "view": str(best["view"][row]),
                        "switched": bool(switched[row]),
                        "base": int(best["base_pred"][row]),
                        "candidate": int(best["candidate_pred"][row]),
                        "truth": int(best["truth"][row]),
                    }
                )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_cache", type=Path, required=True)
    parser.add_argument("--test_cache", type=Path, required=True)
    parser.add_argument("--oof_features", type=Path, required=True)
    parser.add_argument("--projector_checkpoint", type=Path, required=True)
    parser.add_argument("--trajectory_npz", type=Path, required=True)
    parser.add_argument("--node_csv", type=Path, required=True)
    parser.add_argument("--train_profile", type=Path, required=True)
    parser.add_argument("--normality_model", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--review_quantile", type=float, default=0.95)
    parser.add_argument("--knn", type=int, default=15)
    parser.add_argument("--broken_penalty", type=float, default=2.0)
    parser.add_argument("--min_utility_delta", type=float, default=0.0)
    parser.add_argument("--backbone", default="dinov2_vits14")
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--projector_epochs", type=int, default=14)
    parser.add_argument("--projector_lr", type=float, default=1.0e-3)
    parser.add_argument("--projector_batch_size", type=int, default=128)
    parser.add_argument("--tile_drop_rate", type=float, default=0.25)
    parser.add_argument("--concentration_limit", type=float, default=0.65)
    parser.add_argument("--dino_batch_size", type=int, default=48)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--projector_seed", type=int, default=230)
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--device", default="auto")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
