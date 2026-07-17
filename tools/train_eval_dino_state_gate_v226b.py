#!/usr/bin/env python3
"""OOF four-state detector for independent DINO Parent/Fine evidence."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

from src.dual_line import DinoSpatialFeatures
from src.dual_line.decision.joint_gate_v216 import (
    JOINT_STATES,
    JointGateInputV216,
    JointGateSpecV216,
    JointGateV216,
    joint_gate_loss_v216,
)
from src.dual_line.representation import (
    DINO_OBSERVER_GATE_FEATURE_NAMES,
    ConceptLayout,
    DinoConceptEvidenceProjector,
    DinoObserverRelationAdapter,
    project_wave_affinity,
)
from tools.train_eval_dino_concept_evidence_v226 import _evaluate, _load, _loss


PARENT_FEATURES = (
    "parent_confidence",
    "parent_margin",
    "parent_entropy",
    "parent_cls_confidence",
    "parent_tile_confidence",
    "parent_cls_tile_gap",
    "parent_attention_concentration",
    "parent_attention_entropy",
    "parent_mask_flip_rate",
    "parent_mask_prob_drop_max",
)
FINE_FEATURES = tuple(name.replace("parent", "fine") for name in PARENT_FEATURES)
CROSS_FEATURES = (
    "parent_fine_implied_agreement",
    "parent_fine_attention_cosine",
    "parent_fine_attention_overlap",
    "parent_fine_attention_l1",
    "fine_minus_parent_confidence",
    "fine_minus_parent_concentration",
    "fine_minus_parent_flip_rate",
    "fine_implied_parent_support",
    "parent_pred_support_gap",
    "shared_high_confidence",
)
STATE_INDEX = {name: i for i, name in enumerate(JOINT_STATES)}


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _train_projector_fold(
    cache: dict[str, Any],
    layout: ConceptLayout,
    training: np.ndarray,
    *,
    args: argparse.Namespace,
    fold: int,
    device: torch.device,
) -> DinoConceptEvidenceProjector:
    torch.manual_seed(args.seed + fold)
    model = DinoConceptEvidenceProjector(
        embedding_dim=int(cache["cls_token"].shape[-1]),
        parent_classes=len(layout.parent_classes),
        fine_classes=len(layout.fine_classes),
        hidden_dim=args.projector_hidden_dim,
        dropout=args.projector_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.projector_lr, weight_decay=1.0e-4)
    dataset = TensorDataset(
        torch.as_tensor(cache["cls_token"])[training],
        torch.as_tensor(cache["tile_tokens"])[training],
        torch.as_tensor(cache["label"], dtype=torch.long)[training],
    )
    loader = DataLoader(dataset, batch_size=args.projector_batch_size, shuffle=True, num_workers=0)
    for _ in range(args.projector_epochs):
        model.train()
        for cls_token, tile_tokens, fine_target in loader:
            cls_token = cls_token.to(device=device, dtype=torch.float32)
            tile_tokens = tile_tokens.to(device=device, dtype=torch.float32)
            fine_target = fine_target.to(device)
            parent_target = layout.parent_targets(fine_target)
            loss, _ = _loss(
                model,
                cls_token,
                tile_tokens,
                fine_target,
                parent_target,
                drop_rate=args.tile_drop_rate,
                concentration_limit=args.concentration_limit,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return model


def _merge_oof(parts: list[dict[str, np.ndarray]], folds: list[np.ndarray]) -> dict[str, np.ndarray]:
    keys = parts[0].keys()
    merged = {key: np.concatenate([part[key] for part in parts], axis=0) for key in keys}
    merged["fold"] = np.concatenate(
        [np.full(len(indices), fold, dtype=np.int64) for fold, indices in enumerate(folds)], axis=0
    )
    order = np.argsort(merged["index"])
    return {key: value[order] for key, value in merged.items()}


def _load_final_projector(
    checkpoint_path: Path,
    layout: ConceptLayout,
    device: torch.device,
) -> DinoConceptEvidenceProjector:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = DinoConceptEvidenceProjector(
        embedding_dim=int(config["embedding_dim"]),
        parent_classes=len(layout.parent_classes),
        fine_classes=len(layout.fine_classes),
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["state"], strict=True)
    return model


@torch.no_grad()
def _relation_features(
    cache: dict[str, Any],
    wave_path: Path,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    wave = np.load(wave_path, allow_pickle=False)
    wave_names = [str(x) for x in wave["name"].tolist()]
    by_name = {name: i for i, name in enumerate(wave_names)}
    order = np.asarray([by_name[str(name)] for name in cache["name"].tolist()], dtype=np.int64)
    relation = wave["relation"][order].astype(np.float32)
    feature_names = [str(x) for x in wave["feature_names"].tolist()]
    adapter = DinoObserverRelationAdapter().to(device)
    outputs: list[np.ndarray] = []
    for start in range(0, len(order), batch_size):
        stop = min(len(order), start + batch_size)
        cls_token = torch.as_tensor(cache["cls_token"][start:stop], device=device, dtype=torch.float32)
        tile_tokens = torch.as_tensor(cache["tile_tokens"][start:stop], device=device, dtype=torch.float32)
        wave_tensor = torch.as_tensor(relation[start:stop], device=device, dtype=torch.float32)
        wave_affinity, asymmetry = project_wave_affinity(wave_tensor, feature_names)
        spatial = DinoSpatialFeatures(
            cls_token=cls_token,
            patch_tokens=torch.empty((stop - start, 0, cls_token.shape[-1]), device=device),
            tile_tokens=tile_tokens,
            patch_grid=(16, 16),
            tile_grid=(4, 4),
        )
        out = adapter(
            spatial,
            wave_affinity=wave_affinity,
            wave_directional_asymmetry=asymmetry,
        )
        outputs.append(out.gate_features.cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def _probability_features(probability: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sorted_prob = np.sort(probability, axis=1)
    confidence = sorted_prob[:, -1]
    margin = sorted_prob[:, -1] - sorted_prob[:, -2]
    entropy = -(probability * np.log(np.maximum(probability, 1.0e-8))).sum(axis=1) / np.log(probability.shape[1])
    return confidence.astype(np.float32), margin.astype(np.float32), entropy.astype(np.float32)


def _attention_entropy(attention: np.ndarray) -> np.ndarray:
    return (
        -(attention * np.log(np.maximum(attention, 1.0e-8))).sum(axis=1) / np.log(attention.shape[1])
    ).astype(np.float32)


def _state_features(
    packet: dict[str, np.ndarray],
    relation_features: np.ndarray,
    layout: ConceptLayout,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    parent_prob = packet["parent_prob"].astype(np.float32)
    fine_prob = packet["fine_prob"].astype(np.float32)
    parent_conf, parent_margin, parent_entropy = _probability_features(parent_prob)
    fine_conf, fine_margin, fine_entropy = _probability_features(fine_prob)
    row = np.arange(len(parent_prob))
    parent_pred = packet["parent_pred"].astype(np.int64)
    fine_pred = packet["fine_pred"].astype(np.int64)
    parent_cls_conf = packet["parent_cls_prob"][row, parent_pred].astype(np.float32)
    parent_tile_conf = packet["parent_tile_prob"][row, parent_pred].astype(np.float32)
    fine_cls_conf = packet["fine_cls_prob"][row, fine_pred].astype(np.float32)
    fine_tile_conf = packet["fine_tile_prob"][row, fine_pred].astype(np.float32)
    parent_attention = packet["parent_pred_attention"].astype(np.float32)
    fine_attention = packet["fine_pred_attention"].astype(np.float32)
    parent_concentration = packet["parent_attention_concentration"].astype(np.float32)
    fine_concentration = packet["fine_attention_concentration"].astype(np.float32)
    parent_flip = packet["parent_mask_flip_rate"].astype(np.float32)
    fine_flip = packet["fine_mask_flip_rate"].astype(np.float32)
    parent_features = np.stack(
        [
            parent_conf,
            parent_margin,
            parent_entropy,
            parent_cls_conf,
            parent_tile_conf,
            np.abs(parent_cls_conf - parent_tile_conf),
            parent_concentration,
            _attention_entropy(parent_attention),
            parent_flip,
            packet["parent_mask_prob_drop_max"].astype(np.float32),
        ],
        axis=1,
    )
    fine_features = np.stack(
        [
            fine_conf,
            fine_margin,
            fine_entropy,
            fine_cls_conf,
            fine_tile_conf,
            np.abs(fine_cls_conf - fine_tile_conf),
            fine_concentration,
            _attention_entropy(fine_attention),
            fine_flip,
            packet["fine_mask_prob_drop_max"].astype(np.float32),
        ],
        axis=1,
    )
    fine_implied_parent = np.asarray(layout.fine_to_parent_index, dtype=np.int64)[fine_pred]
    dot = (parent_attention * fine_attention).sum(axis=1)
    norm = np.linalg.norm(parent_attention, axis=1) * np.linalg.norm(fine_attention, axis=1) + 1.0e-8
    implied_support = parent_prob[row, fine_implied_parent]
    predicted_support = parent_prob[row, parent_pred]
    cross_features = np.stack(
        [
            (parent_pred == fine_implied_parent).astype(np.float32),
            dot / norm,
            np.minimum(parent_attention, fine_attention).sum(axis=1),
            np.abs(parent_attention - fine_attention).mean(axis=1),
            fine_conf - parent_conf,
            fine_concentration - parent_concentration,
            fine_flip - parent_flip,
            implied_support,
            predicted_support - implied_support,
            parent_conf * fine_conf,
        ],
        axis=1,
    ).astype(np.float32)
    return parent_features.astype(np.float32), fine_features.astype(np.float32), cross_features, relation_features


def _state_targets(packet: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parent_valid = packet["parent_pred"].astype(np.int64) == packet["parent_label"].astype(np.int64)
    fine_valid = packet["fine_pred"].astype(np.int64) == packet["fine_label"].astype(np.int64)
    state = np.full(len(parent_valid), STATE_INDEX["neither"], dtype=np.int64)
    state[parent_valid & fine_valid] = STATE_INDEX["both_valid"]
    state[parent_valid & ~fine_valid] = STATE_INDEX["parent_only"]
    state[~parent_valid & fine_valid] = STATE_INDEX["fine_only"]
    return parent_valid.astype(np.float32), fine_valid.astype(np.float32), state


def _balanced_weights(target: np.ndarray) -> np.ndarray:
    counts = np.bincount(target.astype(np.int64), minlength=2).astype(np.float32)
    weights = len(target) / np.maximum(2.0 * counts, 1.0)
    return weights[target.astype(np.int64)]


def _metrics(output: Any, parent_target: np.ndarray, fine_target: np.ndarray, state_target: np.ndarray) -> dict[str, Any]:
    parent_probability = output.parent_validity.detach().cpu().numpy()
    fine_probability = output.fine_validity.detach().cpu().numpy()
    parent_pred = parent_probability >= 0.5
    fine_pred = fine_probability >= 0.5
    state_pred = np.full(len(parent_pred), STATE_INDEX["neither"], dtype=np.int64)
    state_pred[parent_pred & fine_pred] = STATE_INDEX["both_valid"]
    state_pred[parent_pred & ~fine_pred] = STATE_INDEX["parent_only"]
    state_pred[~parent_pred & fine_pred] = STATE_INDEX["fine_only"]
    invalid = state_target != STATE_INDEX["both_valid"]
    review = state_pred != STATE_INDEX["both_valid"]
    return {
        "n": int(len(state_target)),
        "parent_valid_auc": float(roc_auc_score(parent_target, parent_probability)) if len(np.unique(parent_target)) == 2 else None,
        "fine_valid_auc": float(roc_auc_score(fine_target, fine_probability)) if len(np.unique(fine_target)) == 2 else None,
        "state_accuracy": float((state_pred == state_target).mean()),
        "state_macro_f1": float(f1_score(state_target, state_pred, labels=np.arange(4), average="macro", zero_division=0)),
        "state_counts": {name: int(np.sum(state_target == i)) for i, name in enumerate(JOINT_STATES)},
        "state_pred_counts": {name: int(np.sum(state_pred == i)) for i, name in enumerate(JOINT_STATES)},
        "state_recall": {
            name: float(np.mean(state_pred[state_target == i] == i)) if np.any(state_target == i) else None
            for i, name in enumerate(JOINT_STATES)
        },
        "confusion_matrix": confusion_matrix(state_target, state_pred, labels=np.arange(4)).tolist(),
        "review_rate": float(review.mean()),
        "any_invalid_recall": float(np.mean(review[invalid])) if invalid.any() else None,
        "both_valid_false_review_rate": float(np.mean(review[~invalid])) if (~invalid).any() else None,
    }


def _model_input(features: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], indices: np.ndarray, device: torch.device) -> JointGateInputV216:
    return JointGateInputV216(
        sample_keys=[str(x) for x in indices.tolist()],
        parent_features=torch.as_tensor(features[0][indices], device=device),
        fine_features=torch.as_tensor(features[1][indices], device=device),
        cross_features=torch.as_tensor(features[2][indices], device=device),
        observation_features=torch.as_tensor(features[3][indices], device=device),
    )


def run(args: argparse.Namespace) -> None:
    device = _device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_cache = _load(args.train_cache)
    test_cache = _load(args.test_cache)
    fine_classes = [str(x) for x in train_cache["class_names"].tolist()]
    parent_map = json.loads(args.parent_map.read_text(encoding="utf-8"))
    layout = ConceptLayout.from_parent_map(fine_classes, parent_map)

    labels = train_cache["label"].astype(np.int64)
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    heldout_folds: list[np.ndarray] = []
    oof_parts: list[dict[str, np.ndarray]] = []
    for fold, (training, heldout) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
        model = _train_projector_fold(train_cache, layout, training, args=args, fold=fold, device=device)
        metrics, arrays = _evaluate(
            model,
            train_cache,
            heldout.astype(np.int64),
            layout,
            device,
            args.eval_batch_size,
            counterfactual=True,
        )
        heldout_folds.append(heldout.astype(np.int64))
        oof_parts.append(arrays)
        print(f"[OOF {fold + 1}/{args.folds}] {json.dumps(metrics)}", flush=True)
    oof_packet = _merge_oof(oof_parts, heldout_folds)

    final_projector = _load_final_projector(args.projector_checkpoint, layout, device)
    test_indices = np.arange(len(test_cache["label"]), dtype=np.int64)
    test_projector_metrics, test_packet = _evaluate(
        final_projector,
        test_cache,
        test_indices,
        layout,
        device,
        args.eval_batch_size,
        counterfactual=True,
    )
    train_relation = _relation_features(train_cache, args.train_wave, device=device, batch_size=args.eval_batch_size)
    test_relation = _relation_features(test_cache, args.test_wave, device=device, batch_size=args.eval_batch_size)
    train_features = _state_features(oof_packet, train_relation, layout)
    test_features = _state_features(test_packet, test_relation, layout)
    parent_target, fine_target, state_target = _state_targets(oof_packet)
    test_parent_target, test_fine_target, test_state_target = _state_targets(test_packet)

    val_indices = np.flatnonzero(oof_packet["fold"].astype(np.int64) == 0)
    train_indices = np.flatnonzero(oof_packet["fold"].astype(np.int64) != 0)
    spec = JointGateSpecV216(
        parent_dim=len(PARENT_FEATURES),
        fine_dim=len(FINE_FEATURES),
        cross_dim=len(CROSS_FEATURES),
        observation_dim=len(DINO_OBSERVER_GATE_FEATURE_NAMES),
        parent_hidden_dim=24,
        fine_hidden_dim=24,
        relation_hidden_dim=32,
        relation_state_dim=20,
        action_hidden_dim=16,
        dropout=0.10,
    )
    gate = JointGateV216(spec).to(device)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=args.gate_lr, weight_decay=1.0e-4)
    parent_weight = _balanced_weights(parent_target[train_indices])
    fine_weight = _balanced_weights(fine_target[train_indices])
    state_counts = np.bincount(state_target[train_indices], minlength=4).astype(np.float32)
    state_class_weight = torch.as_tensor(
        np.sqrt(len(train_indices) / np.maximum(4.0 * state_counts, 1.0)),
        device=device,
        dtype=torch.float32,
    )
    training_dataset = TensorDataset(
        torch.as_tensor(train_indices, dtype=torch.long),
        torch.as_tensor(parent_weight, dtype=torch.float32),
        torch.as_tensor(fine_weight, dtype=torch.float32),
    )
    loader = DataLoader(training_dataset, batch_size=args.gate_batch_size, shuffle=True)
    best_state = None
    best_score = -1.0
    best_epoch = 0
    for epoch in range(1, args.gate_epochs + 1):
        gate.train()
        for idx, parent_sample_weight, fine_sample_weight in loader:
            indices = idx.numpy()
            batch = _model_input(train_features, indices, device)
            output = gate(batch)
            loss, _ = joint_gate_loss_v216(
                output,
                parent_valid_target=torch.as_tensor(parent_target[indices], device=device),
                fine_valid_target=torch.as_tensor(fine_target[indices], device=device),
                state_target=torch.as_tensor(state_target[indices], device=device),
                parent_weight=1.0,
                fine_weight=1.0,
                state_weight=0.35,
                parent_sample_weight=parent_sample_weight.to(device),
                fine_sample_weight=fine_sample_weight.to(device),
                state_class_weight=state_class_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(), 5.0)
            optimizer.step()
        if epoch % 5 == 0 or epoch == args.gate_epochs:
            gate.eval()
            with torch.no_grad():
                output = gate(_model_input(train_features, val_indices, device))
            metrics = _metrics(
                output,
                parent_target[val_indices],
                fine_target[val_indices],
                state_target[val_indices],
            )
            auc_parent = metrics["parent_valid_auc"] or 0.0
            auc_fine = metrics["fine_valid_auc"] or 0.0
            score = float(auc_parent + auc_fine + metrics["state_macro_f1"])
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = {key: value.detach().cpu() for key, value in gate.state_dict().items()}
    assert best_state is not None
    gate.load_state_dict(best_state)
    gate.eval()
    with torch.no_grad():
        oof_output = gate(_model_input(train_features, np.arange(len(state_target)), device))
        test_output = gate(_model_input(test_features, np.arange(len(test_state_target)), device))
    oof_metrics = _metrics(oof_output, parent_target, fine_target, state_target)
    test_metrics = _metrics(test_output, test_parent_target, test_fine_target, test_state_target)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state": best_state,
            "spec": asdict(spec),
            "feature_names": {
                "parent": PARENT_FEATURES,
                "fine": FINE_FEATURES,
                "cross": CROSS_FEATURES,
                "observation": DINO_OBSERVER_GATE_FEATURE_NAMES,
            },
            "layout": {
                "fine_classes": layout.fine_classes,
                "parent_classes": layout.parent_classes,
                "fine_to_parent_index": layout.fine_to_parent_index,
            },
        },
        args.out_dir / "dino_state_gate_v226b.pt",
    )
    np.savez(
        args.out_dir / "oof_state_features.npz",
        name=np.asarray(train_cache["name"]),
        fold=oof_packet["fold"],
        parent_features=train_features[0],
        fine_features=train_features[1],
        cross_features=train_features[2],
        observation_features=train_features[3],
        parent_valid=parent_target,
        fine_valid=fine_target,
        state_target=state_target,
    )
    summary = {
        "mode": "v226b_oof_dino_four_state_detector",
        "uses_gate_for_prediction_switch": False,
        "runtime_uses_truth": False,
        "truth_usage": "OOF state supervision and final TEST audit only",
        "test_used_for_threshold_or_model_selection": False,
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "projector_folds": args.folds,
        "projector_epochs_per_fold": args.projector_epochs,
        "test_projector": test_projector_metrics,
        "oof_detector": oof_metrics,
        "test_detector": test_metrics,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_cache", type=Path, required=True)
    parser.add_argument("--test_cache", type=Path, required=True)
    parser.add_argument("--train_wave", type=Path, required=True)
    parser.add_argument("--test_wave", type=Path, required=True)
    parser.add_argument("--projector_checkpoint", type=Path, required=True)
    parser.add_argument("--parent_map", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--projector_epochs", type=int, default=8)
    parser.add_argument("--projector_hidden_dim", type=int, default=160)
    parser.add_argument("--projector_dropout", type=float, default=0.10)
    parser.add_argument("--projector_lr", type=float, default=1.0e-3)
    parser.add_argument("--projector_batch_size", type=int, default=128)
    parser.add_argument("--tile_drop_rate", type=float, default=0.25)
    parser.add_argument("--concentration_limit", type=float, default=0.65)
    parser.add_argument("--gate_epochs", type=int, default=200)
    parser.add_argument("--gate_batch_size", type=int, default=128)
    parser.add_argument("--gate_lr", type=float, default=1.0e-3)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=226)
    parser.add_argument("--device", default="auto")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
