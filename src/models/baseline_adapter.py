"""
src/models/baseline_adapter.py
==============================
Trainer-facing wrapper for the Phase 4 single-image baseline.

The Phase 6 Trainer and its PhaseController were written against SENSE's
interface:

  * ``model(left, right)`` with a ``model.stereo_enabled`` flag
  * ``model.encoder.left_branch`` — an ``nn.Sequential`` whose ``[0]`` element
    is the visual backbone (used for freezing and for the head/backbone LR split)

``SingleImageClassifier`` (Phase 4) has none of these, so it cannot be trained
by ``scripts/train.py`` as-is. ``SingleImageTrainerModel`` is a thin subclass
that adds exactly that interface and nothing else:

  * ``stereo_enabled = False``   -> the Trainer passes ``right=None``
  * ``forward(x, right=None)``   -> ``right`` is ignored
  * ``.encoder.left_branch``     -> ``branch[0]`` is ``self.backbone``

The branch view is deliberately a plain Python object, not an ``nn.Module``:
registering the backbone a second time would duplicate every key in the state
dict. Because it is not registered, ``state_dict()`` keys stay identical to
``SingleImageClassifier`` (``backbone.*`` / ``classifier.*``), so Phase 4
checkpoints and ``scripts/evaluate_baseline.py`` keep working unchanged.

This wrapper exists only to give the baseline the same *training machinery* as
SENSE (macro-F1 checkpoint selection, validation-only monitoring, the P6.5
experiment record). It does not change the architecture, the data, or the
recipe.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.visual_encoder import SingleImageClassifier

__all__ = ["SingleImageTrainerModel"]


class _MonoBranchView:
    """
    Minimal stand-in for ``BiChannelEncoder`` exposing ``.left_branch``.

    Not an ``nn.Module`` on purpose — see the module docstring. PhaseController
    only ever reads ``branch[0]`` (the backbone), so no module registration is
    needed for it to work, and skipping registration keeps ``state_dict()``
    free of duplicate keys.
    """

    def __init__(self, backbone: nn.Module) -> None:
        self.left_branch = nn.Sequential(backbone, nn.Identity())


class SingleImageTrainerModel(SingleImageClassifier):
    """
    Phase 4 baseline (ResNet18 by default) made trainable by the Phase 6 Trainer.

    Behaviour is identical to ``SingleImageClassifier``; only the attribute
    surface expected by ``Trainer`` / ``PhaseController`` is added.
    """

    stereo_enabled: bool = False

    def __init__(
        self,
        backbone: str = "resnet18",
        num_classes: int = 17,
        pretrained: bool = True,
    ) -> None:
        super().__init__(backbone=backbone, num_classes=num_classes, pretrained=pretrained)
        self.encoder = _MonoBranchView(self.backbone)

    def forward(self, x: torch.Tensor, right: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x:     Left image tensor (B, 3, H, W).
            right: Ignored. Present only so the Trainer's two-argument call works.

        Returns:
            logits: Raw class logits (B, num_classes).
        """
        return super().forward(x)
