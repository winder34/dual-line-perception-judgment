"""Persistent, label-free single-image inference for the Dual-Line demo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.dual_line import BackboneAdapterConfig, build_backbone_adapter, preprocess_roi_image
from tools.eval_dino_guided_reobserve_v227 import EXPANDED_VIEW_NAMES, _view_boxes
from tools.infer_label_free_v235 import (
    _best_unlabeled_candidate,
    _candidate_feature_rows,
    _load_npz,
    _predict_packet,
)
from tools.train_eval_dino_reobserve_promotion_v228 import _device, _layout
from tools.train_eval_error_risk_ranker_v233 import _fused_score
from tools.train_eval_multi_expert_correction_v234 import _combined_score, _predict_expert
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


@dataclass(frozen=True)
class SingleImageArtifacts:
    train_cache: Path
    normality_model: Path
    projector_checkpoint: Path
    trajectory_npz: Path
    node_csv: Path
    risk_model: Path
    correction_model: Path


@dataclass(frozen=True)
class SingleImageSettings:
    backbone: str = "resnet18"
    input_size: int = 224
    observer_grid: int = 4
    review_budget_pct: int = 10
    knn: int = 15
    backbone_batch_size: int = 48
    eval_batch_size: int = 256
    device: str = "auto"


class SingleImageInferenceEngine:
    """Load trained artifacts once and evaluate unlabeled PIL images in memory."""

    def __init__(self, artifacts: SingleImageArtifacts, settings: SingleImageSettings):
        self.artifacts = artifacts
        self.settings = settings
        self.device = _device(settings.device)
        self.train_cache = _load_npz(artifacts.train_cache)
        self.checkpoint = torch.load(
            artifacts.projector_checkpoint, map_location="cpu", weights_only=False
        )
        self.layout = _layout(self.checkpoint)
        self.mapping = np.asarray(self.layout.fine_to_parent_index, dtype=np.int64)
        self.projector = _load_final_projector(self.checkpoint, self.device)
        self.node_features, _, self.centroids = _load_trajectory(
            artifacts.trajectory_npz, artifacts.node_csv
        )
        self.normality = joblib.load(artifacts.normality_model)
        self.risk_model = joblib.load(artifacts.risk_model)
        self.correction = joblib.load(artifacts.correction_model)
        self.adapter = build_backbone_adapter(
            BackboneAdapterConfig(
                backbone=settings.backbone,
                weights="default",
                input_size=settings.input_size,
                device=settings.device,
                preprocess_id=f"imagenet_rgb_{settings.input_size}",
            )
        )

    @torch.no_grad()
    def _spatial_cache(self, image: Image.Image, name: str) -> dict[str, np.ndarray]:
        tensor = preprocess_roi_image(image, (0.0, 0.0, 1.0, 1.0), self.settings.input_size)
        spatial = self.adapter.encode_spatial(
            tensor.unsqueeze(0), observer_grid=self.settings.observer_grid
        )
        return {
            "name": np.asarray([name], dtype=str),
            "class_names": np.asarray(self.train_cache["class_names"], dtype=str),
            "cls_token": spatial.cls_token.detach().cpu().numpy().astype(np.float16),
            "tile_tokens": spatial.tile_tokens.detach().cpu().numpy().astype(np.float16),
        }

    @torch.no_grad()
    def _view_cache(
        self,
        image: Image.Image,
        base_packet: dict[str, np.ndarray],
        name: str,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        boxes = _view_boxes(
            base_packet["parent_pred_attention"][0],
            base_packet["fine_pred_attention"][0],
            policy="expanded",
        )
        tensors: list[torch.Tensor] = []
        views: list[str] = []
        bboxes: list[tuple[float, float, float, float]] = []
        seen: set[tuple[float, float, float, float]] = set()
        for view in EXPANDED_VIEW_NAMES:
            bbox = tuple(float(value) for value in boxes[view])
            key = tuple(round(value, 5) for value in bbox)
            if key in seen:
                continue
            seen.add(key)
            tensors.append(preprocess_roi_image(image, bbox, self.settings.input_size))
            views.append(view)
            bboxes.append(bbox)

        cls_parts: list[np.ndarray] = []
        tile_parts: list[np.ndarray] = []
        for start in range(0, len(tensors), self.settings.backbone_batch_size):
            spatial = self.adapter.encode_spatial(
                torch.stack(tensors[start : start + self.settings.backbone_batch_size]),
                observer_grid=self.settings.observer_grid,
            )
            cls_parts.append(spatial.cls_token.detach().cpu().numpy().astype(np.float16))
            tile_parts.append(spatial.tile_tokens.detach().cpu().numpy().astype(np.float16))
        return (
            {
                "name": np.asarray([f"{name}::{view}" for view in views], dtype=str),
                "cls_token": np.concatenate(cls_parts),
                "tile_tokens": np.concatenate(tile_parts),
            },
            {
                "source_index": np.zeros(len(views), dtype=np.int64),
                "view": np.asarray(views, dtype=str),
                "bbox": np.asarray(bboxes, dtype=np.float32),
            },
        )

    @staticmethod
    def _probability_details(classes: tuple[str, ...], probability: np.ndarray) -> list[dict[str, Any]]:
        order = np.argsort(probability)[::-1][:3]
        return [
            {"class": classes[int(index)], "probability": float(probability[int(index)])}
            for index in order
        ]

    @torch.no_grad()
    def predict(self, image: Image.Image, name: str = "uploaded_image") -> dict[str, Any]:
        image = image.convert("RGB")
        cache = self._spatial_cache(image, name)
        base_packet = _predict_packet(
            self.projector,
            cache,
            np.asarray([0], dtype=np.int64),
            self.device,
            self.settings.eval_batch_size,
        )
        profile, _, _ = _profile(
            cache["tile_tokens"], base_packet, self.node_features, self.centroids
        )
        base_reduced, base_normality_risk = _normality_transform(profile, self.normality)
        base_knn = _knn_features(
            cache["cls_token"],
            base_packet["fine_pred"],
            base_packet["parent_pred"],
            self.train_cache["cls_token"],
            self.train_cache["label"],
            self.mapping,
            source_index=None,
            k=self.settings.knn,
        )
        base_features = _base_features(
            base_reduced, base_normality_risk, base_packet, base_knn
        )
        direct = self.risk_model["direct_model"].predict_proba(base_features)[:, 1]
        parent_risk = self.risk_model["parent_model"].predict_proba(base_features)[:, 1]
        fine_risk = self.risk_model["fine_model"].predict_proba(base_features)[:, 1]
        risk_score = _fused_score(
            direct,
            parent_risk,
            fine_risk,
            base_normality_risk,
            self.risk_model["fusion"],
            references=self.risk_model["score_references"],
        )
        threshold_key = f"train_budget_{self.settings.review_budget_pct}pct"
        review_threshold = float(self.risk_model["deployment_thresholds"][threshold_key])
        review = bool(risk_score[0] >= review_threshold)

        base_expert = _predict_expert(self.correction["base_models"], base_features)
        base_score = _combined_score(base_expert, self.correction["joint_weight"])
        base_parent = int(base_packet["parent_pred"][0])
        base_fine = int(base_packet["fine_pred"][0])
        final_parent = base_parent
        final_fine = base_fine
        switched = False
        chosen_view = "keep"
        chosen_bbox: list[float] | None = None
        utility: float | None = None
        candidate_score_value: float | None = None
        candidates: list[dict[str, Any]] = []

        if review:
            view_cache, metadata = self._view_cache(image, base_packet, name)
            candidate_packet = _predict_packet(
                self.projector,
                view_cache,
                np.arange(len(view_cache["name"]), dtype=np.int64),
                self.device,
                self.settings.eval_batch_size,
            )
            candidate_profile, _, _ = _profile(
                view_cache["tile_tokens"], candidate_packet, self.node_features, self.centroids
            )
            candidate_reduced, candidate_risk = _normality_transform(
                candidate_profile, self.normality
            )
            candidate_knn = _knn_features(
                view_cache["cls_token"],
                candidate_packet["fine_pred"],
                candidate_packet["parent_pred"],
                self.train_cache["cls_token"],
                self.train_cache["label"],
                self.mapping,
                source_index=None,
                k=self.settings.knn,
            )
            group = _view_group_features(candidate_packet, metadata["source_index"])
            candidate_features = _candidate_features(
                base_packet=base_packet,
                candidate_packet=candidate_packet,
                metadata=metadata,
                base_reduced=base_reduced,
                candidate_reduced=candidate_reduced,
                base_risk=base_normality_risk,
                candidate_risk=candidate_risk,
                base_knn=base_knn,
                candidate_knn=candidate_knn,
                group=group,
                layout=self.layout,
            )
            rows = _candidate_feature_rows(
                base_packet, candidate_packet, metadata, candidate_features
            )
            candidate_expert = _predict_expert(
                self.correction["candidate_models"], candidate_features
            )
            candidate_scores = _combined_score(
                candidate_expert, self.correction["joint_weight"]
            )
            best = _best_unlabeled_candidate(
                rows,
                candidate_scores,
                base_score,
                max_weight=self.correction["pair_consensus_max_weight"],
                support_bonus=self.correction["pair_consensus_support_bonus"],
            )
            best_row = int(best["row"][0])
            candidate_score_value = float(best["candidate_score"][0])
            utility = candidate_score_value - float(best["base_score"][0])
            changed = bool(
                best["candidate_parent"][0] != best["base_parent"][0]
                or best["candidate_fine"][0] != best["base_fine"][0]
            )
            switched = bool(changed and utility >= self.correction["utility_threshold"])
            chosen_view = str(best["view"][0])
            chosen_bbox = metadata["bbox"][best_row].astype(float).tolist()
            if switched:
                final_parent = int(best["candidate_parent"][0])
                final_fine = int(best["candidate_fine"][0])

            for row in np.argsort(candidate_scores)[::-1]:
                candidates.append(
                    {
                        "view": str(metadata["view"][row]),
                        "bbox": metadata["bbox"][row].astype(float).tolist(),
                        "parent": self.layout.parent_classes[int(candidate_packet["parent_pred"][row])],
                        "fine": self.layout.fine_classes[int(candidate_packet["fine_pred"][row])],
                        "score": float(candidate_scores[row]),
                        "parent_validity": float(candidate_expert["parent"][row]),
                        "fine_validity": float(candidate_expert["fine"][row]),
                        "joint_validity": float(candidate_expert["joint"][row]),
                        "selected": bool(row == best_row),
                    }
                )

        return {
            "input": {"name": name, "width": image.width, "height": image.height},
            "prediction": {
                "parent": self.layout.parent_classes[final_parent],
                "fine": self.layout.fine_classes[final_fine],
                "base_parent": self.layout.parent_classes[base_parent],
                "base_fine": self.layout.fine_classes[base_fine],
                "parent_top3": self._probability_details(
                    self.layout.parent_classes, F.softmax(torch.as_tensor(base_packet["parent_prob"][0]), dim=0).numpy()
                    if not np.isclose(base_packet["parent_prob"][0].sum(), 1.0) else base_packet["parent_prob"][0]
                ),
                "fine_top3": self._probability_details(
                    self.layout.fine_classes, F.softmax(torch.as_tensor(base_packet["fine_prob"][0]), dim=0).numpy()
                    if not np.isclose(base_packet["fine_prob"][0].sum(), 1.0) else base_packet["fine_prob"][0]
                ),
            },
            "gate": {
                "review_requested": review,
                "switched": switched,
                "decision": "SWITCH" if switched else ("REVIEW_KEEP" if review else "KEEP"),
                "risk_score": float(risk_score[0]),
                "risk_threshold": review_threshold,
                "direct_risk": float(direct[0]),
                "parent_risk": float(parent_risk[0]),
                "fine_risk": float(fine_risk[0]),
                "normality_risk": float(base_normality_risk[0]),
                "base_validity": float(base_score[0]),
                "candidate_validity": candidate_score_value,
                "utility": utility,
                "utility_threshold": float(self.correction["utility_threshold"]),
                "chosen_view": chosen_view,
                "chosen_bbox": chosen_bbox,
            },
            "candidates": candidates,
            "label_free": True,
        }
