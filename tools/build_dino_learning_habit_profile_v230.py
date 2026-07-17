#!/usr/bin/env python3
"""Build runtime Parent/Fine habit profiles from a learned DINO trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.dual_line.representation import ConceptLayout, DinoConceptEvidenceProjector
from tools.train_eval_dino_concept_evidence_v226 import _load


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1.0e-8)


def _snapshot_index(labels: np.ndarray, requested: str, fallback: int) -> int:
    found = np.flatnonzero(labels.astype(str) == requested)
    return int(found[0]) if len(found) else int(fallback)


def _adoption_index(curve: np.ndarray) -> np.ndarray:
    target = curve[-1] * 0.5
    adopted = np.zeros(curve.shape[1], dtype=np.float32)
    for node in range(curve.shape[1]):
        found = np.flatnonzero(curve[:, node] >= target[node])
        adopted[node] = float(found[0] if len(found) else len(curve) - 1)
    return adopted / max(1, len(curve) - 1)


def _node_features(
    trajectory: dict[str, np.ndarray],
    node_audit: pd.DataFrame,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    labels = trajectory["snapshot_label"].astype(str)
    parent = trajectory["parent_attention_mass"].astype(np.float32)
    fine = trajectory["fine_attention_mass"].astype(np.float32)
    combined = 0.5 * (parent + fine)
    batch4 = _snapshot_index(labels, "epoch_01_batch_004", min(3, len(labels) - 1))
    batch8 = _snapshot_index(labels, "epoch_01_batch_008", min(4, len(labels) - 1))
    batch16 = _snapshot_index(labels, "epoch_01_batch_016", min(5, len(labels) - 1))
    positive_steps = (np.diff(combined, axis=0) > 0).mean(axis=0)
    support_scale = max(1.0, float(node_audit["final_sample_support"].max()))
    class_scale = max(1.0, float(node_audit["class_support"].max()))
    heldout_ratio = node_audit["heldout_support"].to_numpy(np.float32) / np.maximum(
        node_audit["final_sample_support"].to_numpy(np.float32), 1.0
    )
    features = np.column_stack(
        (
            combined[batch4] - combined[0],
            combined[batch8] - combined[0],
            combined[batch16] - combined[0],
            combined[-1] - combined[batch16],
            combined[-1],
            combined.mean(axis=0),
            positive_steps,
            node_audit["epoch_persistence"].to_numpy(np.float32),
            node_audit["final_sample_support"].to_numpy(np.float32) / support_scale,
            heldout_ratio,
            node_audit["class_support"].to_numpy(np.float32) / class_scale,
            node_audit["sample_concentration"].to_numpy(np.float32),
            parent[batch8] - parent[0],
            fine[batch8] - fine[0],
            parent[-1] - fine[-1],
            _adoption_index(parent),
            _adoption_index(fine),
            _adoption_index(parent) - _adoption_index(fine),
        )
    ).astype(np.float32)
    names = [
        "growth_to_batch4",
        "growth_to_batch8",
        "growth_to_batch16",
        "late_growth_after_batch16",
        "final_attention_mass",
        "trajectory_attention_mean",
        "positive_step_ratio",
        "epoch_persistence",
        "sample_support_ratio",
        "heldout_transfer_ratio",
        "class_coverage_ratio",
        "sample_concentration",
        "parent_growth_to_batch8",
        "fine_growth_to_batch8",
        "final_parent_minus_fine",
        "parent_adoption_time",
        "fine_adoption_time",
        "parent_minus_fine_adoption_time",
    ]
    return features, names, labels


def _load_model(checkpoint_path: Path, device: torch.device) -> tuple[DinoConceptEvidenceProjector, ConceptLayout]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    layout_data = checkpoint["layout"]
    layout = ConceptLayout(
        fine_classes=tuple(str(value) for value in layout_data["fine_classes"]),
        parent_classes=tuple(str(value) for value in layout_data["parent_classes"]),
        fine_to_parent_index=tuple(int(value) for value in layout_data["fine_to_parent_index"]),
    )
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
    return model, layout


def _node_distribution(attention: np.ndarray, assignment: np.ndarray, nodes: int) -> np.ndarray:
    result = np.zeros((len(attention), nodes), dtype=np.float32)
    rows = np.arange(len(attention))
    for tile in range(assignment.shape[1]):
        np.add.at(result, (rows, assignment[:, tile]), attention[:, tile])
    return result / np.maximum(result.sum(axis=1, keepdims=True), 1.0e-8)


def _weighted_profile(
    attention: np.ndarray,
    assignment: np.ndarray,
    node_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tile_features = node_features[assignment]
    mean = np.einsum("bt,btf->bf", attention, tile_features)
    variance = np.einsum("bt,btf->bf", attention, (tile_features - mean[:, None, :]) ** 2)
    maximum = tile_features.max(axis=1)
    return mean.astype(np.float32), np.sqrt(np.maximum(variance, 0.0)).astype(np.float32), maximum


def run(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = _device(args.device)
    cache = _load(args.target_cache)
    with np.load(args.trajectory_npz, allow_pickle=False) as loaded:
        trajectory = {key: loaded[key] for key in loaded.files}
    audit = pd.read_csv(args.node_csv)
    node_features, node_feature_names, snapshot_labels = _node_features(trajectory, audit)
    centroids = _normalize_rows(trajectory["node_centroid"].astype(np.float32))
    model, layout = _load_model(args.projector_checkpoint, device)

    packet: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "parent_pred", "fine_pred", "parent_conf", "fine_conf",
            "parent_attention", "fine_attention",
        )
    }
    cls_tokens = cache["cls_token"].astype(np.float32)
    tile_tokens = cache["tile_tokens"].astype(np.float32)
    for start in range(0, len(cls_tokens), args.batch_size):
        stop = min(len(cls_tokens), start + args.batch_size)
        with torch.inference_mode():
            output = model(
                torch.as_tensor(cls_tokens[start:stop], device=device),
                torch.as_tensor(tile_tokens[start:stop], device=device),
            )
            parent_prob = output.parent_logits.softmax(dim=-1)
            fine_prob = output.fine_logits.softmax(dim=-1)
            parent_pred = parent_prob.argmax(dim=-1)
            fine_pred = fine_prob.argmax(dim=-1)
            row = torch.arange(stop - start, device=device)
            packet["parent_pred"].append(parent_pred.cpu().numpy())
            packet["fine_pred"].append(fine_pred.cpu().numpy())
            packet["parent_conf"].append(parent_prob.max(dim=-1).values.cpu().numpy())
            packet["fine_conf"].append(fine_prob.max(dim=-1).values.cpu().numpy())
            packet["parent_attention"].append(
                output.parent_spatial_attention[row, parent_pred].cpu().numpy()
            )
            packet["fine_attention"].append(
                output.fine_spatial_attention[row, fine_pred].cpu().numpy()
            )
    values = {key: np.concatenate(parts).astype(np.float32) for key, parts in packet.items()}
    values["parent_pred"] = values["parent_pred"].astype(np.int64)
    values["fine_pred"] = values["fine_pred"].astype(np.int64)

    normalized_tiles = _normalize_rows(tile_tokens.reshape(-1, tile_tokens.shape[-1]))
    assignment = (normalized_tiles @ centroids.T).argmax(axis=1).reshape(tile_tokens.shape[:2])
    parent_dist = _node_distribution(values["parent_attention"], assignment, len(centroids))
    fine_dist = _node_distribution(values["fine_attention"], assignment, len(centroids))
    parent_mean, parent_std, parent_max = _weighted_profile(
        values["parent_attention"], assignment, node_features
    )
    fine_mean, fine_std, fine_max = _weighted_profile(
        values["fine_attention"], assignment, node_features
    )
    midpoint = 0.5 * (parent_dist + fine_dist)
    parent_kl = np.sum(parent_dist * np.log((parent_dist + 1.0e-8) / (midpoint + 1.0e-8)), axis=1)
    fine_kl = np.sum(fine_dist * np.log((fine_dist + 1.0e-8) / (midpoint + 1.0e-8)), axis=1)
    js_distance = 0.5 * (parent_kl + fine_kl)
    distribution_l1 = np.abs(parent_dist - fine_dist).sum(axis=1)
    distribution_agreement = np.sum(parent_dist * fine_dist, axis=1) / np.maximum(
        np.linalg.norm(parent_dist, axis=1) * np.linalg.norm(fine_dist, axis=1), 1.0e-8
    )
    profile = np.column_stack(
        (
            parent_mean, parent_std, parent_max,
            fine_mean, fine_std, fine_max,
            parent_mean - fine_mean,
            np.abs(parent_mean - fine_mean),
            js_distance, distribution_l1, distribution_agreement,
            values["parent_conf"], values["fine_conf"],
        )
    ).astype(np.float32)
    profile_names: list[str] = []
    for prefix in (
        "parent_mean", "parent_std", "parent_max", "fine_mean", "fine_std", "fine_max",
        "parent_minus_fine", "parent_fine_abs_gap",
    ):
        profile_names.extend(f"{prefix}__{name}" for name in node_feature_names)
    profile_names.extend(
        ("node_distribution_js", "node_distribution_l1", "node_distribution_cosine",
         "parent_confidence", "fine_confidence")
    )

    np.savez_compressed(
        args.out_dir / "learning_habit_profile_v230.npz",
        sample_key=np.asarray(cache["name"]).astype(str),
        parent_prediction=values["parent_pred"],
        fine_prediction=values["fine_pred"],
        parent_confidence=values["parent_conf"],
        fine_confidence=values["fine_conf"],
        tile_node_assignment=assignment.astype(np.int16),
        parent_node_distribution=parent_dist,
        fine_node_distribution=fine_dist,
        node_habit_features=node_features,
        node_habit_feature_names=np.asarray(node_feature_names),
        habit_profile=profile,
        habit_profile_feature_names=np.asarray(profile_names),
        node_centroid=centroids,
    )
    node_frame = pd.DataFrame(node_features, columns=node_feature_names)
    node_frame.insert(0, "node", audit["node"].astype(str))
    node_frame.to_csv(args.out_dir / "node_learning_habit_profiles.csv", index=False)

    summary: dict[str, Any] = {
        "mode": "v230c_dino_learning_habit_profile",
        "n_samples": int(len(profile)),
        "n_nodes": int(len(centroids)),
        "node_feature_count": len(node_feature_names),
        "sample_profile_feature_count": len(profile_names),
        "runtime_profile_contains_labels": False,
        "snapshot_labels": snapshot_labels.tolist(),
        "mean_parent_fine_node_js": float(js_distance.mean()),
        "mean_parent_fine_node_cosine": float(distribution_agreement.mean()),
        "artifacts": {
            "runtime_profile": str(args.out_dir / "learning_habit_profile_v230.npz"),
            "node_profiles": str(args.out_dir / "node_learning_habit_profiles.csv"),
        },
    }
    if "label" in cache:
        fine_label = cache["label"].astype(np.int64)
        parent_mapping = np.asarray(layout.fine_to_parent_index, dtype=np.int64)
        parent_label = parent_mapping[fine_label]
        summary["label_audit_only"] = {
            "fine_accuracy": float(np.mean(values["fine_pred"] == fine_label)),
            "parent_accuracy": float(np.mean(values["parent_pred"] == parent_label)),
            "both_correct": int(
                np.sum((values["fine_pred"] == fine_label) & (values["parent_pred"] == parent_label))
            ),
            "parent_only_correct": int(
                np.sum((values["fine_pred"] != fine_label) & (values["parent_pred"] == parent_label))
            ),
            "fine_only_correct": int(
                np.sum((values["fine_pred"] == fine_label) & (values["parent_pred"] != parent_label))
            ),
            "neither_correct": int(
                np.sum((values["fine_pred"] != fine_label) & (values["parent_pred"] != parent_label))
            ),
        }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory_npz", type=Path, required=True)
    parser.add_argument("--node_csv", type=Path, required=True)
    parser.add_argument("--projector_checkpoint", type=Path, required=True)
    parser.add_argument("--target_cache", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
