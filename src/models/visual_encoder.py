"""
src/models/visual_encoder.py
Phase 4 — SingleImageClassifier: ResNet18 backbone + linear 17-way head.
Left-image only baseline for pipeline-correctness verification.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn


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
