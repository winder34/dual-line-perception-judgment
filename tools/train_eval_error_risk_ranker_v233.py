#!/usr/bin/env python3
"""Train a TRAIN-OOF error-risk ranker and audit fixed review budgets.

This module only detects samples that deserve review. It intentionally does not
generate reobservation candidates or approve prediction changes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tools.train_eval_dino_concept_evidence_v226 import _load
from tools.train_eval_dino_concept_evidence_v226 import _evaluate
from tools.train_eval_dino_reobserve_promotion_v228 import _layout
from tools.train_eval_trajectory_guided_defer_v231 import (
    _base_features,
    _knn_features,
    _load_final_projector,
    _load_trajectory,
    _normality_transform,
    _profile,
)
from tools.train_eval_dino_reobserve_promotion_v228 import _device


BUDGETS = (0.05, 0.10, 0.20)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    factory: Callable[[int], Any]


def _weights(target: np.ndarray) -> np.ndarray:
    target = target.astype(bool)
    positive = int(target.sum())
    negative = int((~target).sum())
    result = np.ones(len(target), dtype=np.float32)
    if positive and negative:
        result[target] = len(target) / (2.0 * positive)
        result[~target] = len(target) / (2.0 * negative)
    return result


def _specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            "logreg_c0.2",
            lambda seed: make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=0.2,
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ),
        ModelSpec(
            "logreg_c1.0",
            lambda seed: make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=1.0,
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ),
        ModelSpec(
            "hgb_d2_l20",
            lambda seed: HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=180,
                max_depth=2,
                min_samples_leaf=20,
                l2_regularization=2.0,
                random_state=seed,
            ),
        ),
        ModelSpec(
            "hgb_d3_l20",
            lambda seed: HistGradientBoostingClassifier(
                learning_rate=0.04,
                max_iter=220,
                max_depth=3,
                min_samples_leaf=20,
                l2_regularization=3.0,
                random_state=seed,
            ),
        ),
        ModelSpec(
            "hgb_d3_l35",
            lambda seed: HistGradientBoostingClassifier(
                learning_rate=0.04,
                max_iter=220,
                max_depth=3,
                min_samples_leaf=35,
                l2_regularization=5.0,
                random_state=seed,
            ),
        ),
    ]


def _fit(model: Any, features: np.ndarray, target: np.ndarray) -> Any:
    if isinstance(model, HistGradientBoostingClassifier):
        return model.fit(features, target, sample_weight=_weights(target))
    return model.fit(features, target)


def _rank01(score: np.ndarray) -> np.ndarray:
    order = np.argsort(score, kind="stable")
    rank = np.empty(len(score), dtype=np.float32)
    rank[order] = np.arange(len(score), dtype=np.float32)
    return rank / max(1, len(score) - 1)


def _percentile_from_reference(score: np.ndarray, reference: np.ndarray) -> np.ndarray:
    sorted_reference = np.sort(reference.astype(np.float32))
    return (
        np.searchsorted(sorted_reference, score, side="right").astype(np.float32)
        / max(1, len(sorted_reference))
    )


def _budget_metrics(target: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    target = target.astype(bool)
    order = np.argsort(score)[::-1]
    total_error = int(target.sum())
    output: dict[str, Any] = {
        "n": int(len(target)),
        "errors": total_error,
        "roc_auc": float(roc_auc_score(target, score)) if 0 < total_error < len(target) else None,
        "average_precision": float(average_precision_score(target, score)) if total_error else None,
    }
    weighted_recall = 0.0
    for weight, budget in zip((0.5, 0.3, 0.2), BUDGETS):
        count = max(1, int(np.ceil(len(target) * budget)))
        reviewed = order[:count]
        captured = int(target[reviewed].sum())
        recall = captured / max(1, total_error)
        precision = captured / count
        output[f"review_{int(budget * 100)}pct"] = {
            "count": count,
            "captured": captured,
            "recall": float(recall),
            "precision": float(precision),
        }
        weighted_recall += weight * recall
    output["weighted_budget_recall"] = float(weighted_recall)

    # Risk-coverage audit: reject high-risk samples first and average retained risk.
    accepted_error = total_error
    retained_risks: list[float] = []
    for rejected, index in enumerate(order):
        retained = len(target) - rejected
        if retained:
            retained_risks.append(accepted_error / retained)
        accepted_error -= int(target[index])
    output["aurc"] = float(np.mean(retained_risks))
    return output


def _selection_key(metrics: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(metrics["weighted_budget_recall"]),
        float(metrics["average_precision"] or 0.0),
        -float(metrics["aurc"]),
    )


def _threshold_metrics(
    target: np.ndarray, score: np.ndarray, thresholds: dict[str, float]
) -> dict[str, Any]:
    target = target.astype(bool)
    result: dict[str, Any] = {}
    for name, threshold in thresholds.items():
        reviewed = score >= threshold
        captured = int(np.sum(reviewed & target))
        count = int(reviewed.sum())
        result[name] = {
            "threshold": float(threshold),
            "review_count": count,
            "review_rate": float(count / len(target)),
            "captured": captured,
            "recall": float(captured / max(1, int(target.sum()))),
            "precision": float(captured / max(1, count)),
        }
    return result


def _train_target(
    features: np.ndarray,
    target: np.ndarray,
    folds: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, Any, dict[str, Any]]:
    trials: list[dict[str, Any]] = []
    winner: tuple[tuple[float, float, float], ModelSpec, np.ndarray, dict[str, Any]] | None = None
    for spec_index, spec in enumerate(_specs()):
        score = np.zeros(len(target), dtype=np.float32)
        for fold in sorted(np.unique(folds).tolist()):
            training = folds != fold
            heldout = folds == fold
            model = spec.factory(seed + spec_index * 20 + int(fold))
            _fit(model, features[training], target[training])
            score[heldout] = model.predict_proba(features[heldout])[:, 1].astype(np.float32)
        metrics = _budget_metrics(target, score)
        trials.append({"model": spec.name, **metrics})
        rank = _selection_key(metrics)
        if winner is None or rank > winner[0]:
            winner = (rank, spec, score, metrics)
    assert winner is not None
    _, spec, oof_score, metrics = winner
    final_model = spec.factory(seed + 1000)
    _fit(final_model, features, target)
    return oof_score, final_model, {
        "selected_model": spec.name,
        "selected_metrics": metrics,
        "trials": trials,
    }


def _fused_score(
    direct: np.ndarray,
    parent: np.ndarray,
    fine: np.ndarray,
    normality: np.ndarray,
    config: dict[str, Any],
    references: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    if references is None:
        direct_rank = _rank01(direct)
        parent_rank = _rank01(parent)
        fine_rank = _rank01(fine)
        normality_rank = _rank01(normality)
    else:
        direct_rank = _percentile_from_reference(direct, references["direct"])
        parent_rank = _percentile_from_reference(parent, references["parent"])
        fine_rank = _percentile_from_reference(fine, references["fine"])
        normality_rank = _percentile_from_reference(normality, references["normality"])
    branch = np.maximum(parent_rank, fine_rank)
    supervised = config["direct_weight"] * direct_rank + (1.0 - config["direct_weight"]) * branch
    return (
        (1.0 - config["normality_weight"]) * supervised
        + config["normality_weight"] * normality_rank
    ).astype(np.float32)


def _select_fusion(
    target: np.ndarray,
    direct: np.ndarray,
    parent: np.ndarray,
    fine: np.ndarray,
    normality: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    trials: list[dict[str, Any]] = []
    winner: tuple[tuple[float, float, float], dict[str, Any], np.ndarray] | None = None
    for direct_weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        for normality_weight in (0.0, 0.15, 0.30, 0.50):
            config = {
                "direct_weight": direct_weight,
                "normality_weight": normality_weight,
            }
            score = _fused_score(direct, parent, fine, normality, config)
            metrics = _budget_metrics(target, score)
            trials.append({**config, **metrics})
            rank = _selection_key(metrics)
            if winner is None or rank > winner[0]:
                winner = (rank, config, score)
    assert winner is not None
    return winner[1], winner[2], trials


def _test_features(args: argparse.Namespace, expected_dim: int) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    train_cache = _load(args.train_cache)
    test_cache = _load(args.test_cache)
    checkpoint = torch.load(args.projector_checkpoint, map_location="cpu", weights_only=False)
    layout = _layout(checkpoint)
    device = _device(args.device)
    model = _load_final_projector(checkpoint, device)
    indices = np.arange(len(test_cache["label"]), dtype=np.int64)
    _, packet = _evaluate(
        model,
        test_cache,
        indices,
        layout,
        device,
        args.eval_batch_size,
        counterfactual=False,
    )
    packet["name"] = test_cache["name"]
    node_features, _, centroids = _load_trajectory(args.trajectory_npz, args.node_csv)
    profile, _, _ = _profile(
        test_cache["tile_tokens"], packet, node_features, centroids
    )
    normality = joblib.load(args.normality_model)
    reduced, risk = _normality_transform(profile, normality)
    mapping = np.asarray(layout.fine_to_parent_index, dtype=np.int64)
    knn = _knn_features(
        test_cache["cls_token"].astype(np.float32),
        packet["fine_pred"].astype(np.int64),
        packet["parent_pred"].astype(np.int64),
        train_cache["cls_token"].astype(np.float32),
        train_cache["label"].astype(np.int64),
        mapping,
        source_index=None,
        k=args.knn,
    )
    features = _base_features(reduced, risk, packet, knn)
    if features.shape[1] != expected_dim:
        raise ValueError(
            f"OOF/test feature dimension mismatch: {expected_dim} != {features.shape[1]}"
        )
    return features, packet, risk


def run(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.oof_base_rows, allow_pickle=False) as loaded:
        oof = {key: loaded[key] for key in loaded.files}
    features = oof["features"].astype(np.float32)
    folds = oof["fold"].astype(np.int64)
    parent_target = (oof["parent_target"].astype(np.int64) == 0).astype(np.int64)
    fine_target = (oof["fine_target"].astype(np.int64) == 0).astype(np.int64)
    any_target = ((parent_target == 1) | (fine_target == 1)).astype(np.int64)
    risk_column = features.shape[1] - 12
    normality_oof = features[:, risk_column]

    direct_oof, direct_model, direct_report = _train_target(
        features, any_target, folds, args.seed
    )
    parent_oof, parent_model, parent_report = _train_target(
        features, parent_target, folds, args.seed + 100
    )
    fine_oof, fine_model, fine_report = _train_target(
        features, fine_target, folds, args.seed + 200
    )
    fusion, fused_oof, fusion_trials = _select_fusion(
        any_target, direct_oof, parent_oof, fine_oof, normality_oof
    )
    deployment_thresholds = {
        f"train_budget_{int(budget * 100)}pct": float(
            np.quantile(fused_oof, 1.0 - budget)
        )
        for budget in BUDGETS
    }

    test_features, packet, test_normality = _test_features(args, features.shape[1])
    direct_test = direct_model.predict_proba(test_features)[:, 1].astype(np.float32)
    parent_test = parent_model.predict_proba(test_features)[:, 1].astype(np.float32)
    fine_test = fine_model.predict_proba(test_features)[:, 1].astype(np.float32)
    fused_test = _fused_score(
        direct_test,
        parent_test,
        fine_test,
        test_normality,
        fusion,
        references={
            "direct": direct_oof,
            "parent": parent_oof,
            "fine": fine_oof,
            "normality": normality_oof,
        },
    )
    parent_wrong = packet["parent_pred"].astype(np.int64) != packet["parent_label"].astype(np.int64)
    fine_wrong = packet["fine_pred"].astype(np.int64) != packet["fine_label"].astype(np.int64)
    any_wrong = parent_wrong | fine_wrong

    summary = {
        "mode": "v233a_train_oof_error_risk_ranker",
        "test_used_for_training_or_model_selection": False,
        "correction_or_switch_executed": False,
        "seed": args.seed,
        "feature_dim": int(features.shape[1]),
        "train_oof_counts": {
            "n": int(len(any_target)),
            "any_invalid": int(any_target.sum()),
            "parent_invalid": int(parent_target.sum()),
            "fine_invalid": int(fine_target.sum()),
        },
        "models": {
            "any_invalid": direct_report,
            "parent_invalid": parent_report,
            "fine_invalid": fine_report,
        },
        "fusion": {
            **fusion,
            "oof_metrics": _budget_metrics(any_target, fused_oof),
            "deployment_thresholds": deployment_thresholds,
            "trial_count": len(fusion_trials),
        },
        "baseline_normality": {
            "oof": _budget_metrics(any_target, normality_oof),
            "test": _budget_metrics(any_wrong, test_normality),
        },
        "test": {
            "any_invalid": _budget_metrics(any_wrong, fused_test),
            "single_image_threshold_audit": _threshold_metrics(
                any_wrong, fused_test, deployment_thresholds
            ),
            "parent_invalid": _budget_metrics(parent_wrong, parent_test),
            "fine_invalid": _budget_metrics(fine_wrong, fine_test),
        },
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    joblib.dump(
        {
            "direct_model": direct_model,
            "parent_model": parent_model,
            "fine_model": fine_model,
            "fusion": fusion,
            "feature_dim": int(features.shape[1]),
            "risk_column": int(risk_column),
            "deployment_thresholds": deployment_thresholds,
            "score_references": {
                "direct": np.sort(direct_oof),
                "parent": np.sort(parent_oof),
                "fine": np.sort(fine_oof),
                "normality": np.sort(normality_oof),
            },
        },
        args.out_dir / "error_risk_ranker_v233.pkl",
    )
    np.savez_compressed(
        args.out_dir / "oof_error_risk_audit_v233.npz",
        source_index=oof["source_index"].astype(np.int64),
        fold=folds,
        parent_wrong=parent_target.astype(bool),
        fine_wrong=fine_target.astype(bool),
        any_wrong=any_target.astype(bool),
        direct_score=direct_oof,
        parent_score=parent_oof,
        fine_score=fine_oof,
        normality_score=normality_oof,
        fused_score=fused_oof,
    )
    np.savez_compressed(
        args.out_dir / "test_error_risk_audit_v233.npz",
        sample_key=packet["name"],
        parent_wrong=parent_wrong,
        fine_wrong=fine_wrong,
        any_wrong=any_wrong,
        direct_score=direct_test,
        parent_score=parent_test,
        fine_score=fine_test,
        normality_score=test_normality,
        fused_score=fused_test,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof_base_rows", type=Path, required=True)
    parser.add_argument("--train_cache", type=Path, required=True)
    parser.add_argument("--test_cache", type=Path, required=True)
    parser.add_argument("--normality_model", type=Path, required=True)
    parser.add_argument("--projector_checkpoint", type=Path, required=True)
    parser.add_argument("--trajectory_npz", type=Path, required=True)
    parser.add_argument("--node_csv", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--knn", type=int, default=15)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=233)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
