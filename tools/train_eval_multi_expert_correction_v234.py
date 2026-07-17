#!/usr/bin/env python3
"""Calibrated multi-expert joint correction for v233-reviewed samples."""

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
from sklearn.linear_model import LogisticRegression

from src.dual_line import BackboneAdapterConfig, build_backbone_adapter
from tools.train_eval_dino_concept_evidence_v226 import _evaluate, _load
from tools.train_eval_dino_reobserve_promotion_v228 import (
    _cross_validated_scores,
    _device,
    _layout,
    _make_views,
)
from tools.train_eval_trajectory_guided_defer_v231 import (
    _base_features,
    _candidate_features,
    _candidate_rows,
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


def _calibrate_scores(
    score: np.ndarray,
    target: np.ndarray,
    folds: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, LogisticRegression]:
    calibrated = np.zeros(len(score), dtype=np.float32)
    for fold in sorted(np.unique(folds).tolist()):
        training = folds != fold
        heldout = folds == fold
        model = LogisticRegression(
            C=0.5,
            max_iter=2000,
            class_weight="balanced",
            random_state=seed + int(fold),
        )
        model.fit(score[training, None], target[training])
        calibrated[heldout] = model.predict_proba(score[heldout, None])[:, 1]
    final = LogisticRegression(
        C=0.5,
        max_iter=2000,
        class_weight="balanced",
        random_state=seed + 100,
    )
    final.fit(score[:, None], target)
    return calibrated, final


def _expert_scores(
    features: np.ndarray,
    parent_target: np.ndarray,
    fine_target: np.ndarray,
    folds: np.ndarray,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    joint_target = ((parent_target == 1) & (fine_target == 1)).astype(np.int64)
    raw_parent, parent_model = _cross_validated_scores(
        features, parent_target, folds, seed
    )
    raw_fine, fine_model = _cross_validated_scores(
        features, fine_target, folds, seed + 10
    )
    raw_joint, joint_model = _cross_validated_scores(
        features, joint_target, folds, seed + 20
    )
    parent, parent_calibrator = _calibrate_scores(
        raw_parent, parent_target, folds, seed + 30
    )
    fine, fine_calibrator = _calibrate_scores(
        raw_fine, fine_target, folds, seed + 40
    )
    joint, joint_calibrator = _calibrate_scores(
        raw_joint, joint_target, folds, seed + 50
    )
    return (
        {"parent": parent, "fine": fine, "joint": joint},
        {
            "parent_model": parent_model,
            "fine_model": fine_model,
            "joint_model": joint_model,
            "parent_calibrator": parent_calibrator,
            "fine_calibrator": fine_calibrator,
            "joint_calibrator": joint_calibrator,
        },
    )


def _predict_expert(models: dict[str, Any], features: np.ndarray) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for branch in ("parent", "fine", "joint"):
        raw = models[f"{branch}_model"].predict_proba(features)[:, 1]
        result[branch] = models[f"{branch}_calibrator"].predict_proba(raw[:, None])[:, 1]
    return result


def _combined_score(scores: dict[str, np.ndarray], joint_weight: float) -> np.ndarray:
    branch = np.sqrt(np.maximum(scores["parent"] * scores["fine"], 0.0))
    return (joint_weight * scores["joint"] + (1.0 - joint_weight) * branch).astype(
        np.float32
    )


def _best_candidate(
    rows: dict[str, np.ndarray],
    score: np.ndarray,
    base_score: np.ndarray,
    *,
    max_weight: float,
    support_bonus: float,
) -> dict[str, np.ndarray]:
    chosen: list[int] = []
    aggregate_scores: list[float] = []
    support_counts: list[int] = []
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
            member_score = score[member]
            aggregate = (
                max_weight * float(member_score.max())
                + (1.0 - max_weight) * float(member_score.mean())
                + support_bonus * len(member) / len(candidates)
            )
            representative = int(member[np.argmax(member_score)])
            candidate = (aggregate, len(member), representative)
            if winner is None or candidate > winner:
                winner = candidate
        assert winner is not None
        aggregate_scores.append(winner[0])
        support_counts.append(winner[1])
        chosen.append(winner[2])
    row = np.asarray(chosen, dtype=np.int64)
    source = rows["source_index"][row].astype(np.int64)
    return {
        "row": row,
        "source_index": source,
        "candidate_score": np.asarray(aggregate_scores, dtype=np.float32),
        "base_score": base_score[source],
        "utility_delta": score[row] - base_score[source],
        "base_parent": rows["base_parent_pred"][row].astype(np.int64),
        "base_fine": rows["base_fine_pred"][row].astype(np.int64),
        "candidate_parent": rows["candidate_parent_pred"][row].astype(np.int64),
        "candidate_fine": rows["candidate_fine_pred"][row].astype(np.int64),
        "true_parent": rows["true_parent"][row].astype(np.int64),
        "true_fine": rows["true_fine"][row].astype(np.int64),
        "view": rows["view"][row],
        "pair_support_count": np.asarray(support_counts, dtype=np.int64),
    }


def _policy(best: dict[str, np.ndarray], threshold: float) -> dict[str, Any]:
    changed = (best["candidate_parent"] != best["base_parent"]) | (
        best["candidate_fine"] != best["base_fine"]
    )
    switch = changed & (best["utility_delta"] >= threshold)
    final_parent = np.where(switch, best["candidate_parent"], best["base_parent"])
    final_fine = np.where(switch, best["candidate_fine"], best["base_fine"])
    base_correct = (best["base_parent"] == best["true_parent"]) & (
        best["base_fine"] == best["true_fine"]
    )
    final_correct = (final_parent == best["true_parent"]) & (
        final_fine == best["true_fine"]
    )
    fixed = int(np.sum(~base_correct & final_correct))
    broken = int(np.sum(base_correct & ~final_correct))
    return {
        "threshold": float(threshold),
        "switch_count": int(switch.sum()),
        "fixed": fixed,
        "broken": broken,
        "wrong_to_wrong": int(np.sum(switch & ~base_correct & ~final_correct)),
        "net": fixed - broken,
    }


def _select_policy(
    base_scores: dict[str, np.ndarray],
    candidate_scores: dict[str, np.ndarray],
    rows: dict[str, np.ndarray],
    broken_penalty: float,
) -> tuple[float, float, float, float, dict[str, Any]]:
    winner: tuple[
        tuple[float, int, int, float, float],
        float,
        float,
        float,
        float,
        dict[str, Any],
    ] | None = None
    for joint_weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        base = _combined_score(base_scores, joint_weight)
        candidate = _combined_score(candidate_scores, joint_weight)
        for max_weight in (0.5, 0.75, 1.0):
            for support_bonus in (0.0, 0.05, 0.10, 0.20):
                best = _best_candidate(
                    rows,
                    candidate,
                    base,
                    max_weight=max_weight,
                    support_bonus=support_bonus,
                )
                thresholds = np.unique(
                    np.concatenate(
                        (
                            np.asarray([0.0, 1.0], dtype=np.float32),
                            best["utility_delta"][best["utility_delta"] >= 0.0],
                        )
                    )
                )
                for threshold in thresholds:
                    metrics = _policy(best, float(threshold))
                    rank = (
                        metrics["fixed"] - broken_penalty * metrics["broken"],
                        metrics["fixed"],
                        -metrics["broken"],
                        float(joint_weight),
                        float(threshold),
                    )
                    if winner is None or rank > winner[0]:
                        winner = (
                            rank,
                            joint_weight,
                            max_weight,
                            support_bonus,
                            float(threshold),
                            metrics,
                        )
    assert winner is not None
    return winner[1], winner[2], winner[3], winner[4], winner[5]


def _test_feature_packets(
    args: argparse.Namespace,
    checkpoint: dict[str, Any],
    layout: Any,
    device: torch.device,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    train_cache = _load(args.train_cache)
    test_cache = _load(args.test_cache)
    model = _load_final_projector(checkpoint, device)
    indices = np.arange(len(test_cache["label"]), dtype=np.int64)
    _, packet = _evaluate(
        model, test_cache, indices, layout, device, args.eval_batch_size, counterfactual=False
    )
    node_features, _, centroids = _load_trajectory(args.trajectory_npz, args.node_csv)
    profile, _, _ = _profile(test_cache["tile_tokens"], packet, node_features, centroids)
    normality = joblib.load(args.normality_model)
    reduced, risk = _normality_transform(profile, normality)
    mapping = np.asarray(layout.fine_to_parent_index, dtype=np.int64)
    knn = _knn_features(
        test_cache["cls_token"],
        packet["fine_pred"],
        packet["parent_pred"],
        train_cache["cls_token"],
        train_cache["label"],
        mapping,
        source_index=None,
        k=args.knn,
    )
    features = _base_features(reduced, risk, packet, knn)
    return test_cache, packet, features, reduced, risk, knn


def run(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    base_rows = _load_npz(args.oof_base_rows)
    candidate_rows = _load_npz(args.oof_candidate_rows)
    base_scores, base_models = _expert_scores(
        base_rows["features"].astype(np.float32),
        base_rows["parent_target"].astype(np.int64),
        base_rows["fine_target"].astype(np.int64),
        base_rows["fold"].astype(np.int64),
        args.seed,
    )
    candidate_scores, candidate_models = _expert_scores(
        candidate_rows["features"].astype(np.float32),
        candidate_rows["parent_target"].astype(np.int64),
        candidate_rows["fine_target"].astype(np.int64),
        candidate_rows["fold"].astype(np.int64),
        args.seed + 100,
    )
    joint_weight, max_weight, support_bonus, utility_threshold, oof_policy = _select_policy(
        base_scores,
        candidate_scores,
        candidate_rows,
        args.broken_penalty,
    )

    device = _device(args.device)
    checkpoint = torch.load(args.projector_checkpoint, map_location="cpu", weights_only=False)
    layout = _layout(checkpoint)
    mapping = np.asarray(layout.fine_to_parent_index, dtype=np.int64)
    test_cache, packet, test_base_features, base_reduced, base_risk, base_knn = (
        _test_feature_packets(args, checkpoint, layout, device)
    )
    if test_base_features.shape[1] != base_rows["features"].shape[1]:
        raise ValueError("base feature schema mismatch")
    test_base_scores = _predict_expert(base_models, test_base_features)

    risk_artifact = joblib.load(args.risk_model)
    review_threshold = float(
        risk_artifact["deployment_thresholds"][f"train_budget_{args.review_budget_pct}pct"]
    )
    risk_audit = _load_npz(args.test_risk_audit)
    if not np.array_equal(risk_audit["sample_key"].astype(str), test_cache["name"].astype(str)):
        raise ValueError("risk audit/test cache sample order mismatch")
    review = risk_audit["fused_score"].astype(np.float32) >= review_threshold
    sources = np.flatnonzero(review)

    adapter = build_backbone_adapter(
        BackboneAdapterConfig(
            backbone=args.backbone,
            weights="default",
            input_size=args.input_size,
            device=args.device,
            preprocess_id=f"imagenet_rgb_{args.input_size}",
        )
    )
    view_cache, metadata = _make_views(
        test_cache,
        packet,
        sources,
        adapter,
        input_size=args.input_size,
        batch_size=args.backbone_batch_size,
    )
    model = _load_final_projector(checkpoint, device)
    _, candidate_packet = _evaluate(
        model,
        view_cache,
        np.arange(len(view_cache["label"]), dtype=np.int64),
        layout,
        device,
        args.eval_batch_size,
        counterfactual=False,
    )
    node_features, _, centroids = _load_trajectory(args.trajectory_npz, args.node_csv)
    candidate_profile, _, _ = _profile(
        view_cache["tile_tokens"], candidate_packet, node_features, centroids
    )
    normality = joblib.load(args.normality_model)
    candidate_reduced, candidate_risk = _normality_transform(candidate_profile, normality)
    train_cache = _load(args.train_cache)
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
        base_risk=base_risk,
        candidate_risk=candidate_risk,
        base_knn=base_knn,
        candidate_knn=candidate_knn,
        group=group,
        layout=layout,
    )
    if candidate_features.shape[1] != candidate_rows["features"].shape[1]:
        raise ValueError("candidate feature schema mismatch")
    rows = _candidate_rows(
        base_packet=packet,
        candidate_packet=candidate_packet,
        metadata=metadata,
        features=candidate_features,
        folds=np.zeros(len(test_cache["label"]), dtype=np.int64),
    )
    test_candidate_scores = _predict_expert(candidate_models, candidate_features)
    base_combined = _combined_score(test_base_scores, joint_weight)
    candidate_combined = _combined_score(test_candidate_scores, joint_weight)
    best = _best_candidate(
        rows,
        candidate_combined,
        base_combined,
        max_weight=max_weight,
        support_bonus=support_bonus,
    )
    changed = (best["candidate_parent"] != best["base_parent"]) | (
        best["candidate_fine"] != best["base_fine"]
    )
    switch = changed & (best["utility_delta"] >= utility_threshold)

    base_parent = packet["parent_pred"].astype(np.int64)
    base_fine = packet["fine_pred"].astype(np.int64)
    final_parent = base_parent.copy()
    final_fine = base_fine.copy()
    switched_source = best["source_index"][switch]
    final_parent[switched_source] = best["candidate_parent"][switch]
    final_fine[switched_source] = best["candidate_fine"][switch]
    true_fine = test_cache["label"].astype(np.int64)
    true_parent = mapping[true_fine]
    base_correct = (base_parent == true_parent) & (base_fine == true_fine)
    final_correct = (final_parent == true_parent) & (final_fine == true_fine)

    candidate_parent = rows["candidate_parent_pred"]
    candidate_fine = rows["candidate_fine_pred"]
    candidate_true = (candidate_parent == rows["true_parent"]) & (
        candidate_fine == rows["true_fine"]
    )
    oracle_sources = np.unique(rows["source_index"][candidate_true])
    reviewed_wrong = sources[~base_correct[sources]]
    oracle_reviewed_wrong = np.intersect1d(reviewed_wrong, oracle_sources)
    fixed_sources = np.flatnonzero(~base_correct & final_correct)
    broken_sources = np.flatnonzero(base_correct & ~final_correct)
    summary = {
        "mode": "v234a_calibrated_multi_expert_joint_correction",
        "test_used_for_training_or_threshold_selection": False,
        "backbone": args.backbone,
        "review_budget_pct": args.review_budget_pct,
        "review_threshold_from_train_oof": review_threshold,
        "joint_weight": joint_weight,
        "pair_consensus_max_weight": max_weight,
        "pair_consensus_support_bonus": support_bonus,
        "utility_threshold_from_train_oof": utility_threshold,
        "train_oof_policy": oof_policy,
        "test_detection": {
            "review_count": int(review.sum()),
            "base_invalid_total": int((~base_correct).sum()),
            "base_invalid_captured": int((~base_correct & review).sum()),
        },
        "test_candidate_oracle": {
            "reviewed_wrong": int(len(reviewed_wrong)),
            "reviewed_wrong_with_correct_pair": int(len(oracle_reviewed_wrong)),
            "oracle_recovery_rate": float(
                len(oracle_reviewed_wrong) / max(1, len(reviewed_wrong))
            ),
        },
        "test_correction": {
            "switch_count": int(switch.sum()),
            "fixed": int(len(fixed_sources)),
            "broken": int(len(broken_sources)),
            "net": int(len(fixed_sources) - len(broken_sources)),
            "reviewed_wrong_recovery_rate": float(
                len(fixed_sources) / max(1, len(reviewed_wrong))
            ),
            "oracle_selector_recall": float(
                len(np.intersect1d(fixed_sources, oracle_reviewed_wrong))
                / max(1, len(oracle_reviewed_wrong))
            ),
        },
        "full_test": {
            "base_parent_accuracy": float(np.mean(base_parent == true_parent)),
            "final_parent_accuracy": float(np.mean(final_parent == true_parent)),
            "base_fine_accuracy": float(np.mean(base_fine == true_fine)),
            "final_fine_accuracy": float(np.mean(final_fine == true_fine)),
            "base_both_accuracy": float(np.mean(base_correct)),
            "final_both_accuracy": float(np.mean(final_correct)),
        },
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    joblib.dump(
        {
            "base_models": base_models,
            "candidate_models": candidate_models,
            "joint_weight": joint_weight,
            "pair_consensus_max_weight": max_weight,
            "pair_consensus_support_bonus": support_bonus,
            "utility_threshold": utility_threshold,
            "review_threshold": review_threshold,
        },
        args.out_dir / "multi_expert_correction_v234.pkl",
    )
    np.savez_compressed(
        args.out_dir / "test_candidate_scores_v234.npz",
        **rows,
        candidate_parent_score=test_candidate_scores["parent"],
        candidate_fine_score=test_candidate_scores["fine"],
        candidate_joint_score=test_candidate_scores["joint"],
        candidate_combined_score=candidate_combined,
        base_combined_score=base_combined[rows["source_index"]],
    )
    with (args.out_dir / "test_correction_audit.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "source_index",
                "name",
                "view",
                "base_parent",
                "base_fine",
                "candidate_parent",
                "candidate_fine",
                "base_score",
                "candidate_score",
                "utility_delta",
                "pair_support_count",
                "switched",
                "base_correct",
                "final_correct",
            ),
        )
        writer.writeheader()
        for row, source in enumerate(best["source_index"]):
            writer.writerow(
                {
                    "source_index": int(source),
                    "name": str(test_cache["name"][source]),
                    "view": str(best["view"][row]),
                    "base_parent": int(best["base_parent"][row]),
                    "base_fine": int(best["base_fine"][row]),
                    "candidate_parent": int(best["candidate_parent"][row]),
                    "candidate_fine": int(best["candidate_fine"][row]),
                    "base_score": float(best["base_score"][row]),
                    "candidate_score": float(best["candidate_score"][row]),
                    "utility_delta": float(best["utility_delta"][row]),
                    "pair_support_count": int(best["pair_support_count"][row]),
                    "switched": bool(switch[row]),
                    "base_correct": bool(base_correct[source]),
                    "final_correct": bool(final_correct[source]),
                }
            )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof_base_rows", type=Path, required=True)
    parser.add_argument("--oof_candidate_rows", type=Path, required=True)
    parser.add_argument("--train_cache", type=Path, required=True)
    parser.add_argument("--test_cache", type=Path, required=True)
    parser.add_argument("--normality_model", type=Path, required=True)
    parser.add_argument("--projector_checkpoint", type=Path, required=True)
    parser.add_argument("--trajectory_npz", type=Path, required=True)
    parser.add_argument("--node_csv", type=Path, required=True)
    parser.add_argument("--risk_model", type=Path, required=True)
    parser.add_argument("--test_risk_audit", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--review_budget_pct", type=int, default=10)
    parser.add_argument("--broken_penalty", type=float, default=4.0)
    parser.add_argument("--knn", type=int, default=15)
    parser.add_argument("--backbone", default="dinov2_vits14")
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--backbone_batch_size", type=int, default=48)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--device", default="auto")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
