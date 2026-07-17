#!/usr/bin/env python3
"""Train an OOF-only promotion gate for expanded DINO reobservation views."""

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
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier

from src.dual_line import BackboneAdapterConfig, FrozenDinoV2BackboneAdapter, preprocess_roi_image
from src.dual_line.decision.joint_gate_v216 import JointGateSpecV216, JointGateV216
from src.dual_line.representation import ConceptLayout
from tools.eval_dino_guided_reobserve_v227 import EXPANDED_VIEW_NAMES, _view_boxes
from tools.train_eval_dino_concept_evidence_v226 import _evaluate, _load
from tools.train_eval_dino_state_gate_v226b import _model_input, _train_projector_fold


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _layout(checkpoint: dict[str, Any]) -> ConceptLayout:
    data = checkpoint["layout"]
    return ConceptLayout(
        fine_classes=tuple(str(x) for x in data["fine_classes"]),
        parent_classes=tuple(str(x) for x in data["parent_classes"]),
        fine_to_parent_index=tuple(int(x) for x in data["fine_to_parent_index"]),
    )


def _validity_probabilities(
    feature_artifact: dict[str, np.ndarray], checkpoint_path: Path, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    gate = JointGateV216(JointGateSpecV216(**checkpoint["spec"])).to(device)
    gate.load_state_dict(checkpoint["state"], strict=True)
    gate.eval()
    features = (
        feature_artifact["parent_features"],
        feature_artifact["fine_features"],
        feature_artifact["cross_features"],
        feature_artifact["observation_features"],
    )
    with torch.no_grad():
        output = gate(_model_input(features, np.arange(len(feature_artifact["name"])), device))
    return output.parent_validity.cpu().numpy(), output.fine_validity.cpu().numpy()


def _prob_stats(probability: np.ndarray) -> tuple[float, float, float]:
    probability = np.asarray(probability, dtype=np.float32)
    ordered = np.sort(probability)
    confidence = float(ordered[-1])
    margin = float(ordered[-1] - ordered[-2])
    entropy = float(
        -(probability * np.log(np.maximum(probability, 1.0e-8))).sum() / np.log(len(probability))
    )
    return confidence, margin, entropy


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1.0e-8
    return float(np.dot(a, b) / denom)


def _feature_names() -> list[str]:
    names = [
        "parent_validity",
        "fine_validity",
        "base_parent_conf",
        "base_parent_margin",
        "base_parent_entropy",
        "candidate_parent_conf",
        "candidate_parent_margin",
        "candidate_parent_entropy",
        "parent_conf_delta",
        "parent_margin_delta",
        "parent_prob_cosine",
        "parent_prob_l1",
        "parent_changed",
        "base_fine_conf",
        "base_fine_margin",
        "base_fine_entropy",
        "candidate_fine_conf",
        "candidate_fine_margin",
        "candidate_fine_entropy",
        "fine_conf_delta",
        "fine_margin_delta",
        "fine_prob_cosine",
        "fine_prob_l1",
        "fine_changed",
        "base_parent_fine_agreement",
        "candidate_parent_fine_agreement",
        "candidate_implied_parent_support",
        "base_implied_parent_support",
        "bbox_area",
        "bbox_aspect",
        "bbox_center_x",
        "bbox_center_y",
    ]
    return names + [f"view_{name}" for name in EXPANDED_VIEW_NAMES]


def _features(
    *,
    base_parent_prob: np.ndarray,
    base_fine_prob: np.ndarray,
    candidate_parent_prob: np.ndarray,
    candidate_fine_prob: np.ndarray,
    parent_validity: float,
    fine_validity: float,
    bbox: np.ndarray,
    view: str,
    layout: ConceptLayout,
) -> np.ndarray:
    bp_conf, bp_margin, bp_entropy = _prob_stats(base_parent_prob)
    bf_conf, bf_margin, bf_entropy = _prob_stats(base_fine_prob)
    cp_conf, cp_margin, cp_entropy = _prob_stats(candidate_parent_prob)
    cf_conf, cf_margin, cf_entropy = _prob_stats(candidate_fine_prob)
    bp = int(np.argmax(base_parent_prob))
    bf = int(np.argmax(base_fine_prob))
    cp = int(np.argmax(candidate_parent_prob))
    cf = int(np.argmax(candidate_fine_prob))
    mapping = np.asarray(layout.fine_to_parent_index, dtype=np.int64)
    x0, y0, x1, y1 = [float(value) for value in bbox]
    width, height = max(x1 - x0, 1.0e-6), max(y1 - y0, 1.0e-6)
    values = [
        float(parent_validity),
        float(fine_validity),
        bp_conf,
        bp_margin,
        bp_entropy,
        cp_conf,
        cp_margin,
        cp_entropy,
        cp_conf - bp_conf,
        cp_margin - bp_margin,
        _cosine(base_parent_prob, candidate_parent_prob),
        float(np.abs(base_parent_prob - candidate_parent_prob).mean()),
        float(cp != bp),
        bf_conf,
        bf_margin,
        bf_entropy,
        cf_conf,
        cf_margin,
        cf_entropy,
        cf_conf - bf_conf,
        cf_margin - bf_margin,
        _cosine(base_fine_prob, candidate_fine_prob),
        float(np.abs(base_fine_prob - candidate_fine_prob).mean()),
        float(cf != bf),
        float(bp == mapping[bf]),
        float(cp == mapping[cf]),
        float(candidate_parent_prob[mapping[cf]]),
        float(base_parent_prob[mapping[bf]]),
        width * height,
        width / height,
        0.5 * (x0 + x1),
        0.5 * (y0 + y1),
    ]
    values.extend(float(view == name) for name in EXPANDED_VIEW_NAMES)
    return np.asarray(values, dtype=np.float32)


def _make_views(
    cache: dict[str, Any],
    base_packet: dict[str, np.ndarray],
    source_indices: np.ndarray,
    adapter: FrozenDinoV2BackboneAdapter,
    *,
    input_size: int,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    by_source = {int(index): position for position, index in enumerate(base_packet["index"].tolist())}
    tensors: list[torch.Tensor] = []
    sources: list[int] = []
    views: list[str] = []
    boxes_out: list[tuple[float, float, float, float]] = []
    for source in source_indices.tolist():
        position = by_source[int(source)]
        boxes = _view_boxes(
            base_packet["parent_pred_attention"][position],
            base_packet["fine_pred_attention"][position],
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
                sources.append(source)
                views.append(view)
                boxes_out.append(bbox)
    cls_parts: list[np.ndarray] = []
    tile_parts: list[np.ndarray] = []
    for start in range(0, len(tensors), batch_size):
        spatial = adapter.encode_spatial(torch.stack(tensors[start : start + batch_size]), observer_grid=4)
        cls_parts.append(spatial.cls_token.cpu().numpy().astype(np.float16))
        tile_parts.append(spatial.tile_tokens.cpu().numpy().astype(np.float16))
    source_array = np.asarray(sources, dtype=np.int64)
    view_cache = {
        "name": np.asarray([f"{cache['name'][i]}::{view}" for i, view in zip(sources, views)], dtype=str),
        "label": cache["label"][source_array].astype(np.int64),
        "class_names": np.asarray(cache["class_names"], dtype=str),
        "cls_token": np.concatenate(cls_parts, axis=0),
        "tile_tokens": np.concatenate(tile_parts, axis=0),
    }
    metadata = {
        "source_index": source_array,
        "view": np.asarray(views, dtype=str),
        "bbox": np.asarray(boxes_out, dtype=np.float32),
    }
    return view_cache, metadata


def _candidate_rows(
    base_packet: dict[str, np.ndarray],
    candidate_packet: dict[str, np.ndarray],
    metadata: dict[str, np.ndarray],
    *,
    parent_validity: np.ndarray,
    fine_validity: np.ndarray,
    folds: np.ndarray,
    layout: ConceptLayout,
) -> dict[str, np.ndarray]:
    by_source = {int(index): position for position, index in enumerate(base_packet["index"].tolist())}
    features: list[np.ndarray] = []
    parent_target: list[int] = []
    fine_target: list[int] = []
    base_parent_pred: list[int] = []
    base_fine_pred: list[int] = []
    candidate_parent_pred: list[int] = []
    candidate_fine_pred: list[int] = []
    true_parent: list[int] = []
    true_fine: list[int] = []
    for row, source in enumerate(metadata["source_index"].tolist()):
        position = by_source[int(source)]
        bp = int(base_packet["parent_pred"][position])
        bf = int(base_packet["fine_pred"][position])
        cp = int(candidate_packet["parent_pred"][row])
        cf = int(candidate_packet["fine_pred"][row])
        tp = int(base_packet["parent_label"][position])
        tf = int(base_packet["fine_label"][position])
        features.append(
            _features(
                base_parent_prob=base_packet["parent_prob"][position],
                base_fine_prob=base_packet["fine_prob"][position],
                candidate_parent_prob=candidate_packet["parent_prob"][row],
                candidate_fine_prob=candidate_packet["fine_prob"][row],
                parent_validity=float(parent_validity[source]),
                fine_validity=float(fine_validity[source]),
                bbox=metadata["bbox"][row],
                view=str(metadata["view"][row]),
                layout=layout,
            )
        )
        parent_target.append(int(bp != tp and cp == tp))
        fine_target.append(int(bf != tf and cf == tf))
        base_parent_pred.append(bp)
        base_fine_pred.append(bf)
        candidate_parent_pred.append(cp)
        candidate_fine_pred.append(cf)
        true_parent.append(tp)
        true_fine.append(tf)
    sources = metadata["source_index"].astype(np.int64)
    return {
        "features": np.stack(features),
        "source_index": sources,
        "fold": folds[sources].astype(np.int64),
        "view": metadata["view"],
        "bbox": metadata["bbox"],
        "parent_target": np.asarray(parent_target, dtype=np.int64),
        "fine_target": np.asarray(fine_target, dtype=np.int64),
        "base_parent_pred": np.asarray(base_parent_pred, dtype=np.int64),
        "base_fine_pred": np.asarray(base_fine_pred, dtype=np.int64),
        "candidate_parent_pred": np.asarray(candidate_parent_pred, dtype=np.int64),
        "candidate_fine_pred": np.asarray(candidate_fine_pred, dtype=np.int64),
        "true_parent": np.asarray(true_parent, dtype=np.int64),
        "true_fine": np.asarray(true_fine, dtype=np.int64),
    }


def _concat(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.concatenate([part[key] for part in parts], axis=0) for key in parts[0]}


def _weights(target: np.ndarray) -> np.ndarray:
    target = target.astype(bool)
    positive, negative = int(target.sum()), int((~target).sum())
    weights = np.ones(len(target), dtype=np.float32)
    if positive and negative:
        weights[target] = len(target) / (2.0 * positive)
        weights[~target] = len(target) / (2.0 * negative)
    return weights


def _new_model(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=240,
        max_depth=4,
        min_samples_leaf=12,
        l2_regularization=1.0,
        random_state=seed,
    )


def _cross_validated_scores(
    features: np.ndarray, target: np.ndarray, folds: np.ndarray, seed: int
) -> tuple[np.ndarray, HistGradientBoostingClassifier]:
    scores = np.zeros(len(target), dtype=np.float32)
    for fold in sorted(np.unique(folds).tolist()):
        training = folds != fold
        heldout = folds == fold
        model = _new_model(seed + int(fold))
        model.fit(features[training], target[training], sample_weight=_weights(target[training]))
        scores[heldout] = model.predict_proba(features[heldout])[:, 1].astype(np.float32)
    final = _new_model(seed + 100)
    final.fit(features, target, sample_weight=_weights(target))
    return scores, final


def _sample_best(rows: dict[str, np.ndarray], scores: np.ndarray, branch: str) -> dict[str, np.ndarray]:
    selected_rows: list[int] = []
    for source in np.unique(rows["source_index"]):
        candidates = np.flatnonzero(rows["source_index"] == source)
        selected_rows.append(int(candidates[np.argmax(scores[candidates])]))
    indices = np.asarray(selected_rows, dtype=np.int64)
    return {
        "row": indices,
        "source_index": rows["source_index"][indices],
        "score": scores[indices],
        "base_pred": rows[f"base_{branch}_pred"][indices],
        "candidate_pred": rows[f"candidate_{branch}_pred"][indices],
        "truth": rows[f"true_{branch}"][indices],
        "view": rows["view"][indices],
    }


def _policy_metrics(best: dict[str, np.ndarray], threshold: float) -> dict[str, Any]:
    switch = (best["score"] >= threshold) & (best["candidate_pred"] != best["base_pred"])
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
        "base_correct": int(base_correct.sum()),
        "final_correct": int(final_correct.sum()),
    }


def _select_threshold(best: dict[str, np.ndarray], broken_penalty: float) -> tuple[float, dict[str, Any]]:
    candidates = np.unique(np.concatenate(([0.0, 1.000001], best["score"])))
    winner: tuple[tuple[float, int, int, int, float], dict[str, Any]] | None = None
    for threshold in candidates:
        metrics = _policy_metrics(best, float(threshold))
        objective = metrics["fixed"] - broken_penalty * metrics["broken"]
        rank = (
            float(objective),
            int(metrics["fixed"]),
            -int(metrics["broken"]),
            -int(metrics["switch_count"]),
            float(threshold),
        )
        if winner is None or rank > winner[0]:
            winner = (rank, metrics)
    assert winner is not None
    return float(winner[1]["threshold"]), winner[1]


def _test_rows(
    packet: dict[str, np.ndarray],
    views: dict[str, np.ndarray],
    parent_validity: np.ndarray,
    fine_validity: np.ndarray,
    layout: ConceptLayout,
) -> dict[str, np.ndarray]:
    source = views["source_index"].astype(np.int64)
    metadata = {"source_index": source, "view": views["view"], "bbox": views["bbox"]}
    candidate = {
        "parent_prob": views["parent_prob"],
        "fine_prob": views["fine_prob"],
        "parent_pred": views["parent_pred"],
        "fine_pred": views["fine_pred"],
    }
    full_packet = dict(packet)
    full_packet["index"] = np.arange(len(packet["name"]), dtype=np.int64)
    return _candidate_rows(
        full_packet,
        candidate,
        metadata,
        parent_validity=parent_validity,
        fine_validity=fine_validity,
        folds=np.zeros(len(packet["name"]), dtype=np.int64),
        layout=layout,
    )


def run(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _device(args.device)
    train_cache = _load(args.train_cache)
    test_packet = _load(args.test_packet)
    test_views = _load(args.test_views)
    test_validity = _load(args.test_validity)
    oof = _load(args.oof_features)
    projector_checkpoint = torch.load(args.projector_checkpoint, map_location="cpu", weights_only=False)
    layout = _layout(projector_checkpoint)
    parent_validity, fine_validity = _validity_probabilities(oof, args.state_gate_checkpoint, device)
    review = (parent_validity < args.review_threshold) | (fine_validity < args.review_threshold)
    folds = oof["fold"].astype(np.int64)
    adapter = FrozenDinoV2BackboneAdapter(
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
        projector_hidden_dim=int(projector_checkpoint["config"]["hidden_dim"]),
        projector_dropout=float(projector_checkpoint["config"]["dropout"]),
        projector_lr=args.projector_lr,
        projector_batch_size=args.projector_batch_size,
        projector_epochs=args.projector_epochs,
        tile_drop_rate=args.tile_drop_rate,
        concentration_limit=args.concentration_limit,
    )
    train_parts: list[dict[str, np.ndarray]] = []
    for fold in sorted(np.unique(folds).tolist()):
        heldout = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        model = _train_projector_fold(
            train_cache,
            layout,
            training,
            args=projector_args,
            fold=int(fold),
            device=device,
        )
        _, base_packet = _evaluate(
            model,
            train_cache,
            heldout,
            layout,
            device,
            args.eval_batch_size,
            counterfactual=False,
        )
        sources = heldout[review[heldout]]
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
        train_parts.append(
            _candidate_rows(
                base_packet,
                candidate_packet,
                metadata,
                parent_validity=parent_validity,
                fine_validity=fine_validity,
                folds=folds,
                layout=layout,
            )
        )
        print(f"[OOF VIEWS] fold={fold} samples={len(sources)} rows={len(metadata['source_index'])}", flush=True)
    train_rows = _concat(train_parts)
    parent_scores, parent_model = _cross_validated_scores(
        train_rows["features"], train_rows["parent_target"], train_rows["fold"], args.seed
    )
    fine_scores, fine_model = _cross_validated_scores(
        train_rows["features"], train_rows["fine_target"], train_rows["fold"], args.seed + 10
    )
    parent_best = _sample_best(train_rows, parent_scores, "parent")
    fine_best = _sample_best(train_rows, fine_scores, "fine")
    parent_threshold, parent_oof_metrics = _select_threshold(parent_best, args.broken_penalty)
    fine_threshold, fine_oof_metrics = _select_threshold(fine_best, args.broken_penalty)

    test_parent_validity = test_validity["parent_validity_probability"].astype(np.float32)
    test_fine_validity = test_validity["fine_validity_probability"].astype(np.float32)
    test_rows = _test_rows(test_packet, test_views, test_parent_validity, test_fine_validity, layout)
    test_parent_scores = parent_model.predict_proba(test_rows["features"])[:, 1].astype(np.float32)
    test_fine_scores = fine_model.predict_proba(test_rows["features"])[:, 1].astype(np.float32)
    test_parent_best = _sample_best(test_rows, test_parent_scores, "parent")
    test_fine_best = _sample_best(test_rows, test_fine_scores, "fine")
    test_parent_metrics = _policy_metrics(test_parent_best, parent_threshold)
    test_fine_metrics = _policy_metrics(test_fine_best, fine_threshold)

    full_parent = test_packet["parent_pred"].astype(np.int64).copy()
    full_fine = test_packet["fine_pred"].astype(np.int64).copy()
    parent_switch = (test_parent_best["score"] >= parent_threshold) & (
        test_parent_best["candidate_pred"] != test_parent_best["base_pred"]
    )
    fine_switch = (test_fine_best["score"] >= fine_threshold) & (
        test_fine_best["candidate_pred"] != test_fine_best["base_pred"]
    )
    full_parent[test_parent_best["source_index"][parent_switch]] = test_parent_best["candidate_pred"][parent_switch]
    full_fine[test_fine_best["source_index"][fine_switch]] = test_fine_best["candidate_pred"][fine_switch]
    true_parent = test_packet["parent_label"].astype(np.int64)
    true_fine = test_packet["fine_label"].astype(np.int64)
    implied_parent = np.asarray(layout.fine_to_parent_index, dtype=np.int64)[full_fine]
    summary = {
        "mode": "v228_oof_dino_reobserve_promotion_gate",
        "test_used_for_training_or_threshold_selection": False,
        "class_identity_features_used": False,
        "prediction_switch_enabled_for_final_audit": True,
        "train_oof_review_samples": int(review.sum()),
        "train_candidate_rows": int(len(train_rows["features"])),
        "train_positive_candidate_rows": {
            "parent": int(train_rows["parent_target"].sum()),
            "fine": int(train_rows["fine_target"].sum()),
        },
        "oof_policy": {"parent": parent_oof_metrics, "fine": fine_oof_metrics},
        "test_review_samples": int(len(np.unique(test_rows["source_index"]))),
        "test_policy": {"parent": test_parent_metrics, "fine": test_fine_metrics},
        "full_test": {
            "base_parent_accuracy": float(np.mean(test_packet["parent_pred"] == true_parent)),
            "final_parent_accuracy": float(np.mean(full_parent == true_parent)),
            "base_fine_accuracy": float(np.mean(test_packet["fine_pred"] == true_fine)),
            "final_fine_accuracy": float(np.mean(full_fine == true_fine)),
            "base_both_accuracy": float(
                np.mean((test_packet["parent_pred"] == true_parent) & (test_packet["fine_pred"] == true_fine))
            ),
            "final_both_accuracy": float(np.mean((full_parent == true_parent) & (full_fine == true_fine))),
            "base_parent_fine_conflict_count": int(
                np.sum(test_packet["parent_pred"].astype(np.int64) != np.asarray(layout.fine_to_parent_index)[test_packet["fine_pred"].astype(np.int64)])
            ),
            "final_parent_fine_conflict_count": int(np.sum(full_parent != implied_parent)),
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    joblib.dump(
        {
            "parent_model": parent_model,
            "fine_model": fine_model,
            "parent_threshold": parent_threshold,
            "fine_threshold": fine_threshold,
            "feature_names": _feature_names(),
        },
        args.out_dir / "reobserve_promotion_gate_v228.pkl",
    )
    np.savez_compressed(
        args.out_dir / "oof_candidate_training_rows.npz",
        **train_rows,
        parent_oof_score=parent_scores,
        fine_oof_score=fine_scores,
        feature_names=np.asarray(_feature_names(), dtype=str),
    )
    with (args.out_dir / "test_promotion_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["branch", "source_index", "name", "score", "threshold", "view", "switched", "base", "candidate", "truth"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for branch, best, threshold, switch in (
            ("parent", test_parent_best, parent_threshold, parent_switch),
            ("fine", test_fine_best, fine_threshold, fine_switch),
        ):
            for row in range(len(best["source_index"])):
                source = int(best["source_index"][row])
                writer.writerow(
                    {
                        "branch": branch,
                        "source_index": source,
                        "name": str(test_packet["name"][source]),
                        "score": float(best["score"][row]),
                        "threshold": float(threshold),
                        "view": str(best["view"][row]),
                        "switched": bool(switch[row]),
                        "base": int(best["base_pred"][row]),
                        "candidate": int(best["candidate_pred"][row]),
                        "truth": int(best["truth"][row]),
                    }
                )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_cache", type=Path, required=True)
    parser.add_argument("--oof_features", type=Path, required=True)
    parser.add_argument("--state_gate_checkpoint", type=Path, required=True)
    parser.add_argument("--projector_checkpoint", type=Path, required=True)
    parser.add_argument("--test_packet", type=Path, required=True)
    parser.add_argument("--test_views", type=Path, required=True)
    parser.add_argument("--test_validity", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--backbone", default="dinov2_vits14")
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--review_threshold", type=float, default=0.5)
    parser.add_argument("--projector_epochs", type=int, default=8)
    parser.add_argument("--projector_lr", type=float, default=1.0e-3)
    parser.add_argument("--projector_batch_size", type=int, default=128)
    parser.add_argument("--tile_drop_rate", type=float, default=0.25)
    parser.add_argument("--concentration_limit", type=float, default=0.65)
    parser.add_argument("--dino_batch_size", type=int, default=48)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--broken_penalty", type=float, default=2.0)
    parser.add_argument("--projector_seed", type=int, default=226)
    parser.add_argument("--seed", type=int, default=228)
    parser.add_argument("--device", default="auto")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
