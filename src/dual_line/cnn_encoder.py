"""Frozen CNN texture encoder for Dual-Line v0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image


@dataclass(frozen=True)
class CNNEncoderConfig:
    backbone: str = "resnet18"
    weights: str = "default"
    input_size: int = 224
    embedding_dim: int = 512


class FrozenCNNEncoder(nn.Module):
    def __init__(self, config: CNNEncoderConfig):
        super().__init__()
        self.config = config
        self.backbone, self.embedding_dim = _make_backbone(config.backbone, config.weights)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        self.backbone.eval()
        return self.backbone(images)


def _make_backbone(backbone: str, weights_mode: str) -> tuple[nn.Module, int]:
    weights_arg = _weights_arg(backbone, weights_mode)
    if backbone == "resnet18":
        model = models.resnet18(weights=weights_arg)
        embedding_dim = int(model.fc.in_features)
        model.fc = nn.Identity()
        return model, embedding_dim
    if backbone == "resnet50":
        model = models.resnet50(weights=weights_arg)
        embedding_dim = int(model.fc.in_features)
        model.fc = nn.Identity()
        return model, embedding_dim
    if backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights_arg)
        embedding_dim = int(model.classifier[1].in_features)
        model.classifier = nn.Identity()
        return model, embedding_dim
    raise ValueError(f"unsupported backbone: {backbone}")


def _weights_arg(backbone: str, weights_mode: str):
    if weights_mode in ("none", "None", ""):
        return None
    if weights_mode != "default":
        raise ValueError(f"unsupported weights mode: {weights_mode}")
    if backbone == "resnet18":
        return models.ResNet18_Weights.DEFAULT
    if backbone == "resnet50":
        return models.ResNet50_Weights.DEFAULT
    if backbone == "efficientnet_b0":
        return models.EfficientNet_B0_Weights.DEFAULT
    return None


def preprocess_roi_image(image: Image.Image, bbox_norm_xyxy: tuple[float, float, float, float], input_size: int) -> torch.Tensor:
    """Crop normalized bbox from image and return normalized CHW tensor."""

    image = image.convert("RGB")
    width, height = image.size
    x0, y0, x1, y1 = bbox_norm_xyxy
    box = (
        int(round(max(0.0, min(1.0, x0)) * width)),
        int(round(max(0.0, min(1.0, y0)) * height)),
        int(round(max(0.0, min(1.0, x1)) * width)),
        int(round(max(0.0, min(1.0, y1)) * height)),
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        box = (0, 0, width, height)
    crop = image.crop(box).resize((int(input_size), int(input_size)), Image.Resampling.BILINEAR)
    arr = np.asarray(crop, dtype=np.float32) / 255.0
    arr = (arr - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)) / np.asarray(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()
