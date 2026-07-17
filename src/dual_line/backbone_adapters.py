"""Backbone adapter contracts for modular Dual-Line representations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch

from .artifacts.schema import ArtifactManifest, FeatureField
from .cnn_encoder import CNNEncoderConfig, FrozenCNNEncoder


@dataclass(frozen=True, slots=True)
class DinoSpatialFeatures:
    """Spatial DINOv2 representation without relation/gate decisions.

    ``tile_tokens`` are adaptively pooled from the native DINO patch grid so
    they align with the project's observer grid.  Relation builders can then
    combine these semantic tokens with wave and boundary evidence without
    changing the legacy ResNet embedding path.
    """

    cls_token: torch.Tensor
    patch_tokens: torch.Tensor
    tile_tokens: torch.Tensor
    patch_grid: tuple[int, int]
    tile_grid: tuple[int, int]


@dataclass(frozen=True, slots=True)
class BackboneAdapterConfig:
    backbone: str = "resnet18"
    weights: str = "default"
    input_size: int = 224
    device: str = "auto"
    preprocess_id: str = "imagenet_rgb_224"

    @property
    def backbone_id(self) -> str:
        return f"{self.backbone}:{self.weights}:input{self.input_size}"


class BackboneAdapter(Protocol):
    config: BackboneAdapterConfig
    embedding_dim: int

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Return one embedding vector per image."""

    def manifest(self, *, sample_keys: list[str], artifact_path: str | Path) -> ArtifactManifest:
        """Return a manifest for an embedding artifact produced by this adapter."""


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class FrozenTorchvisionBackboneAdapter:
    """Frozen torchvision feature extractor behind a stable adapter contract."""

    def __init__(self, config: BackboneAdapterConfig):
        self.config = config
        encoder_config = CNNEncoderConfig(
            backbone=config.backbone,
            weights=config.weights,
            input_size=int(config.input_size),
        )
        self.device = resolve_device(config.device)
        self.encoder = FrozenCNNEncoder(encoder_config).to(self.device)
        self.encoder.eval()
        self.embedding_dim = int(self.encoder.embedding_dim)

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        self.encoder.eval()
        return self.encoder.encode(images.to(self.device, non_blocking=True))

    @torch.no_grad()
    def encode_spatial(
        self,
        images: torch.Tensor,
        *,
        observer_grid: int | tuple[int, int] = 4,
    ) -> DinoSpatialFeatures:
        """Return global and observer-grid tokens from a frozen CNN feature map.

        ResNet's final convolution map is treated as the native spatial token
        grid. Adaptive pooling preserves the same [B, T, D] contract used by
        DINO observer tokens, allowing downstream trajectory and gate modules
        to compare backbones without backbone-specific branches.
        """

        if images.ndim != 4:
            raise ValueError(f"expected BCHW images, got shape={tuple(images.shape)}")
        if isinstance(observer_grid, int):
            tile_grid = (int(observer_grid), int(observer_grid))
        else:
            tile_grid = (int(observer_grid[0]), int(observer_grid[1]))
        if tile_grid[0] <= 0 or tile_grid[1] <= 0:
            raise ValueError(f"observer_grid must be positive, got {tile_grid}")
        if self.config.backbone not in {"resnet18", "resnet50"}:
            raise ValueError(
                "spatial extraction is currently implemented for ResNet adapters only, "
                f"got {self.config.backbone}"
            )

        model = self.encoder.backbone
        model.eval()
        value = images.to(self.device, non_blocking=True)
        value = model.conv1(value)
        value = model.bn1(value)
        value = model.relu(value)
        value = model.maxpool(value)
        value = model.layer1(value)
        value = model.layer2(value)
        value = model.layer3(value)
        feature_map = model.layer4(value)
        patch_grid = (int(feature_map.shape[-2]), int(feature_map.shape[-1]))
        patch_tokens = feature_map.flatten(2).transpose(1, 2).contiguous()
        cls_token = torch.nn.functional.adaptive_avg_pool2d(feature_map, (1, 1)).flatten(1)
        tile_map = torch.nn.functional.adaptive_avg_pool2d(feature_map, tile_grid)
        tile_tokens = tile_map.flatten(2).transpose(1, 2).contiguous()
        return DinoSpatialFeatures(
            cls_token=cls_token,
            patch_tokens=patch_tokens,
            tile_tokens=tile_tokens,
            patch_grid=patch_grid,
            tile_grid=tile_grid,
        )

    def manifest(self, *, sample_keys: list[str], artifact_path: str | Path) -> ArtifactManifest:
        return ArtifactManifest(
            artifact_type="backbone_embedding",
            artifact_path=str(artifact_path),
            sample_keys=[str(x) for x in sample_keys],
            backbone_id=self.config.backbone_id,
            preprocess_id=self.config.preprocess_id,
            feature_schema=[
                FeatureField(
                    name="embedding",
                    dtype="float32",
                    shape=["N", self.embedding_dim],
                    description="Frozen backbone embedding matrix.",
                )
            ],
            metadata={
                "backbone": self.config.backbone,
                "weights": self.config.weights,
                "input_size": int(self.config.input_size),
                "device": str(self.device),
            },
        )


class FrozenDinoV2BackboneAdapter:
    """Frozen DINOv2 encoder with optional observer-aligned spatial output."""

    def __init__(self, config: BackboneAdapterConfig):
        if config.weights != "default":
            raise ValueError("DINOv2 adapter currently requires weights=default")
        if int(config.input_size) % 14 != 0:
            raise ValueError("DINOv2 input_size must be divisible by patch size 14")
        self.config = config
        self.device = resolve_device(config.device)
        self.encoder = torch.hub.load(
            "facebookresearch/dinov2",
            config.backbone,
            pretrained=True,
            trust_repo=True,
        ).to(self.device)
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.embedding_dim = int(getattr(self.encoder, "embed_dim"))
        self.patch_size = 14

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Return CLS tokens for compatibility with the generic adapter API."""

        self.encoder.eval()
        output = self.encoder.forward_features(images.to(self.device, non_blocking=True))
        return output["x_norm_clstoken"]

    @torch.no_grad()
    def encode_spatial(
        self,
        images: torch.Tensor,
        *,
        observer_grid: int | tuple[int, int] = 4,
    ) -> DinoSpatialFeatures:
        """Return native patch tokens and observer-grid-aligned tile tokens.

        The method performs representation extraction only.  It intentionally
        does not compute pairwise similarity, wave agreement, candidates, or
        gate decisions; those belong to downstream relation/decision modules.
        """

        if images.ndim != 4:
            raise ValueError(f"expected BCHW images, got shape={tuple(images.shape)}")
        height, width = int(images.shape[-2]), int(images.shape[-1])
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"DINOv2 image dimensions must be divisible by {self.patch_size}, "
                f"got {(height, width)}"
            )
        if isinstance(observer_grid, int):
            tile_grid = (int(observer_grid), int(observer_grid))
        else:
            tile_grid = (int(observer_grid[0]), int(observer_grid[1]))
        if tile_grid[0] <= 0 or tile_grid[1] <= 0:
            raise ValueError(f"observer_grid must be positive, got {tile_grid}")

        self.encoder.eval()
        output = self.encoder.forward_features(images.to(self.device, non_blocking=True))
        cls_token = output["x_norm_clstoken"]
        patch_tokens = output["x_norm_patchtokens"]
        patch_grid = (height // self.patch_size, width // self.patch_size)
        expected_tokens = patch_grid[0] * patch_grid[1]
        if int(patch_tokens.shape[1]) != expected_tokens:
            raise RuntimeError(
                "DINOv2 patch-token count does not match the input grid: "
                f"tokens={patch_tokens.shape[1]} grid={patch_grid}"
            )

        batch, _, channels = patch_tokens.shape
        patch_map = patch_tokens.transpose(1, 2).reshape(
            batch,
            channels,
            patch_grid[0],
            patch_grid[1],
        )
        tile_map = torch.nn.functional.adaptive_avg_pool2d(patch_map, tile_grid)
        tile_tokens = tile_map.flatten(2).transpose(1, 2).contiguous()
        return DinoSpatialFeatures(
            cls_token=cls_token,
            patch_tokens=patch_tokens,
            tile_tokens=tile_tokens,
            patch_grid=patch_grid,
            tile_grid=tile_grid,
        )

    def manifest(self, *, sample_keys: list[str], artifact_path: str | Path) -> ArtifactManifest:
        return ArtifactManifest(
            artifact_type="backbone_embedding",
            artifact_path=str(artifact_path),
            sample_keys=[str(x) for x in sample_keys],
            backbone_id=self.config.backbone_id,
            preprocess_id=self.config.preprocess_id,
            feature_schema=[
                FeatureField(
                    name="embedding",
                    dtype="float32",
                    shape=["N", self.embedding_dim],
                    description="Frozen DINOv2 normalized CLS token embedding.",
                )
            ],
            metadata={
                "backbone": self.config.backbone,
                "weights": self.config.weights,
                "input_size": int(self.config.input_size),
                "device": str(self.device),
                "hub_repo": "facebookresearch/dinov2",
                "output": "x_norm_clstoken",
            },
        )


def build_backbone_adapter(config: BackboneAdapterConfig) -> BackboneAdapter:
    if config.backbone in {"resnet18", "resnet50", "efficientnet_b0"}:
        return FrozenTorchvisionBackboneAdapter(config)
    if config.backbone in {"dinov2_vits14", "dinov2_vitb14"}:
        return FrozenDinoV2BackboneAdapter(config)
    raise ValueError(f"unsupported backbone adapter: {config.backbone}")
