"""
src/models/visual_encoder.py

Phase 4 — SingleImageClassifier: ResNet18 backbone + linear 17-way head.
Phase 5 — BiChannelEncoder: EfficientNet-B4 × 2 (shared_weights flag), P5.1.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Phase 4 baseline (must stay unchanged — regression-tested in test_models.py)
# ---------------------------------------------------------------------------


class SingleImageClassifier(nn.Module):
    """
    Single-image transfer learning classifier (Phase 4 Baseline).

    Architecture:
        ResNet18 (ImageNet-pretrained, backbone only) → GlobalAvgPool → Linear(512, num_classes)

    Uses left image only. No stereo, no enhancement, no text guidance.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        num_classes: int = 17,
        pretrained: bool = True,
    ) -> None:
        """
        Args:
            backbone: timm model name (default: 'resnet18').
            num_classes: Number of output classes (default: 17 — 16 gestures + true_neg).
            pretrained: Load ImageNet pretrained weights.
        """
        super().__init__()

        # Load backbone with num_classes=0 to strip the classification head
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,  # removes default FC head; returns feature vector
            global_pool="avg",
        )

        feature_dim = self.backbone.num_features

        # Fresh linear classifier head
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image tensor (B, 3, H, W).

        Returns:
            logits: Raw class logits (B, num_classes).
        """
        features = self.backbone(x)  # (B, feature_dim)
        logits = self.classifier(features)  # (B, num_classes)
        return logits

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (only classifier head trains)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True


# ---------------------------------------------------------------------------
# P5.1 — BiChannelEncoder (EfficientNet-B4 × 2, shared_weights flag)
# ---------------------------------------------------------------------------


class BiChannelEncoder(nn.Module):
    """
    Dual-branch visual encoder for stereo image pairs (P5.1).

    Design decisions (documented here — defensible in the thesis):
    - **EfficientNet-B4 backbone:** Chosen for its strong ImageNet accuracy /
      parameter-count trade-off and timm's first-class support. Native
      input size is 380 × 380, but the model works at any resolution via
      adaptive pooling; `image_size` is set by the Phase 5.8 VRAM probe and
      injected via config — never hardcoded here.
    - **Shared weights (`shared_weights=True`):** The left and right branches
      share all backbone parameters. Rationale: stereo images of the same scene
      differ only in viewpoint shift — identical semantic content — so a shared
      representation is well-motivated and halves the visual-encoder parameter
      count. The flag is kept configurable so an ablation can test unshared
      weights if desired, but the project default (recorded in
      `configs/sense_full.yaml`) is `shared_weights: true`.
    - **Global average pooling** applied by timm (`global_pool='avg'`), giving
      a flat feature vector of size `out_dim` per image.
    - **Stereo mode vs. mono mode:** When `stereo_enabled=True` (default) both
      branches are run and their feature vectors are concatenated → (B, 2·out_dim).
      When `stereo_enabled=False` only the left branch is run → (B, out_dim).
      The downstream modules (SelfAttentionBlock, CrossModalFusion) receive
      whichever tensor size this produces; they must be constructed accordingly.
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b4",
        pretrained: bool = True,
        out_dim: int = 512,
        shared_weights: bool = True,
        stereo_enabled: bool = True,
    ) -> None:
        """
        Args:
            backbone: timm model name.
            pretrained: Load ImageNet pretrained weights.
            out_dim: Feature-vector dimension per branch (after global pooling).
                     Implemented via a linear projection so the actual backbone
                     output size does not constrain this value.
            shared_weights: Whether both branches share backbone parameters.
            stereo_enabled: Whether to run both branches (stereo) or left only (mono).
        """
        super().__init__()
        self.stereo_enabled = stereo_enabled
        self.out_dim = out_dim

        def _make_branch() -> nn.Sequential:
            backbone_net = timm.create_model(
                backbone,
                pretrained=pretrained,
                num_classes=0,
                global_pool="avg",
            )
            feat_dim = backbone_net.num_features
            proj = nn.Linear(feat_dim, out_dim, bias=False)
            return nn.Sequential(backbone_net, proj)

        self.left_branch = _make_branch()

        if stereo_enabled:
            if shared_weights:
                # Share all weights — same module object used for both passes
                self.right_branch = self.left_branch
            else:
                self.right_branch = _make_branch()

    @property
    def feature_dim(self) -> int:
        """Output feature dimension: 2×out_dim if stereo, out_dim if mono."""
        return self.out_dim * 2 if self.stereo_enabled else self.out_dim

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            left:  (B, 3, H, W) — left image tensor.
            right: (B, 3, H, W) — right image tensor. Required if stereo_enabled.

        Returns:
            Concatenated feature tensor:
              - stereo: (B, 2 * out_dim)
              - mono:   (B,     out_dim)
        """
        left_feat = self.left_branch(left)  # (B, out_dim)

        if self.stereo_enabled:
            if right is None:
                raise ValueError("right image is required when stereo_enabled=True")
            right_feat = self.right_branch(right)  # (B, out_dim)
            return torch.cat([left_feat, right_feat], dim=1)  # (B, 2*out_dim)

        return left_feat  # (B, out_dim)
