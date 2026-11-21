"""
src/models/factory.py
=====================
Single entry point for building a model from a merged config dict.

Two callers need this:

  * ``scripts/train.py`` — the Trainer accepts a model instance, so something
    must turn the config into the right architecture.
  * ``scripts/evaluate.py`` (P8.2) — its interface is ``--checkpoint {path}``
    with no ``--config`` flag, and checkpoints embed the full merged config
    (``src/utils/checkpoint.py``). Rebuilding the model therefore goes through
    here, using ``payload["config"]``.

Dispatch is explicit: ``model.architecture`` selects the architecture. It
defaults to ``"sense"`` so every existing SENSE config (and every archived
per-run ``config.yaml``) keeps working without edits.

Known architectures:
  * ``sense``        — full SENSE model (``SENSE.from_config``)
  * ``single_image`` — Phase 4 baseline (``SingleImageTrainerModel``), used by
                       ``configs/baseline_resnet18_trainer.yaml``
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from src.models.baseline_adapter import SingleImageTrainerModel
from src.models.sense import SENSE

__all__ = ["ARCHITECTURES", "build_model"]

ARCHITECTURES: tuple[str, ...] = ("sense", "single_image")


def build_model(cfg: dict[str, Any]) -> nn.Module:
    """
    Build the model described by ``cfg``.

    Args:
        cfg: Full merged config dict (as returned by ``load_merged_config``).

    Returns:
        An ``nn.Module`` ready to hand to ``Trainer``.

    Raises:
        ValueError: If ``model.architecture`` is not a known architecture.
    """
    model_cfg = cfg.get("model", {})
    architecture = str(model_cfg.get("architecture", "sense")).lower()

    if architecture == "sense":
        return SENSE.from_config(cfg)

    if architecture == "single_image":
        data_cfg = cfg.get("data", {})
        num_classes = int(model_cfg.get("num_classes") or data_cfg.get("num_classes", 17))
        return SingleImageTrainerModel(
            backbone=str(model_cfg.get("backbone", "resnet18")),
            num_classes=num_classes,
            pretrained=bool(model_cfg.get("pretrained", True)),
        )

    raise ValueError(
        f"Unknown model.architecture '{architecture}'. Expected one of: {', '.join(ARCHITECTURES)}."
    )
