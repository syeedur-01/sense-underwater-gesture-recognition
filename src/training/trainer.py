"""
src/training/trainer.py
=======================
P6.3 — Three-phase progressive unfreezing controller (PhaseController).

Epoch-boundary-driven `requires_grad` toggling on BiChannelEncoder submodules,
with optimizer parameter-group rebuild at each phase transition.

Phase definitions (read from config, never hardcoded):
  phase1: epochs [1, 10]  — backbone fully frozen, head/text/fusion trainable
  phase2: epochs [11, 30] — last N blocks of backbone unfrozen (lr_scale=0.1)
  phase3: epochs [31, 50] — entire backbone unfrozen (lr_scale=0.01)

The PhaseController is wired into the Trainer (P6.8).  It is defined here so
that the Trainer can import a single module.  The Trainer class itself is
implemented in P6.8 and is a thin shell until then.
"""

from __future__ import annotations

import logging
from typing import Any

import torch.nn as nn
from torch.optim import Optimizer

from src.training.scheduler import build_optimizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _efficientnet_blocks(branch: nn.Sequential) -> list[nn.Module]:
    """
    Return the list of stage/block modules inside an EfficientNet-B4 branch.

    timm's EfficientNet exposes ``model.blocks`` — a ModuleList of stages,
    each being a Sequential of individual MBConv blocks.  We flatten those
    stages into a flat list so that "last N blocks" can be counted from the end.

    The branch is an ``nn.Sequential(backbone, proj_linear)``, so we address
    ``branch[0].blocks``.
    """
    backbone = branch[0]  # timm EfficientNet (nn.Module)
    if hasattr(backbone, "blocks"):
        flat: list[nn.Module] = []
        for stage in backbone.blocks:
            if hasattr(stage, "__iter__"):
                flat.extend(stage)
            else:
                flat.append(stage)
        return flat
    # Fallback: return all direct children as blocks
    return list(backbone.children())


def _set_backbone_requires_grad(branch: nn.Sequential, value: bool) -> None:
    """Toggle requires_grad for every parameter in the backbone part of a branch."""
    backbone = branch[0]
    for p in backbone.parameters():
        p.requires_grad = value


def _set_last_n_blocks_requires_grad(branch: nn.Sequential, n: int | None, value: bool) -> None:
    """
    Set requires_grad=value on the last ``n`` EfficientNet blocks of ``branch``.
    If ``n`` is None, apply to ALL blocks.
    """
    blocks = _efficientnet_blocks(branch)
    if n is None or n >= len(blocks):
        target_blocks = blocks
    else:
        target_blocks = blocks[-n:]
    for blk in target_blocks:
        for p in blk.parameters():
            p.requires_grad = value


# ---------------------------------------------------------------------------
# PhaseController
# ---------------------------------------------------------------------------


class PhaseController:
    """
    Three-phase progressive unfreezing controller (P6.3).

    Reads phase boundaries and lr_scale factors from the config dict
    (``training.phases.phaseN``).  Calls ``step(epoch)`` at the *start* of
    each epoch; on a phase boundary it rebuilds the optimizer parameter groups
    at the new learning rate.

    Args:
        model:    SENSE model (must expose ``.encoder`` = BiChannelEncoder).
        cfg:      Full config dict loaded from YAML.
        optimizer: The live optimizer; will be mutated on phase transitions.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: dict[str, Any],
        optimizer: Optimizer,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.optimizer = optimizer
        self._current_phase: int | None = None  # None = not yet applied

        # Parse phase schedule from config
        phases_cfg = cfg.get("training", {}).get("phases", {})
        self._phases: list[dict[str, Any]] = []
        for name in ("phase1", "phase2", "phase3"):
            if name not in phases_cfg:
                continue
            p = phases_cfg[name]
            self._phases.append(
                {
                    "name": name,
                    "start": int(p["epochs"][0]),
                    "end": int(p["epochs"][1]),
                    "freeze_backbone": bool(p.get("freeze_backbone", True)),
                    "unfreeze_last_n_blocks": p.get("unfreeze_last_n_blocks", None),
                    "lr_scale": float(p.get("lr_scale", 1.0)),
                }
            )
        if not self._phases:
            raise ValueError(
                "No phase definitions found in config under training.phases. "
                "Expected phase1, phase2, phase3."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, epoch: int) -> bool:
        """
        Apply the phase configuration for ``epoch``.

        Called at the START of each epoch (before the forward pass).
        Returns True if a phase transition occurred, False otherwise.

        On a transition:
          1. Set ``requires_grad`` flags on the encoder according to the new phase.
          2. Rebuild the optimizer parameter groups at the new lr_scale x base_lr.
        """
        new_phase_idx = self._phase_index_for_epoch(epoch)
        if new_phase_idx is None:
            logger.warning("Epoch %d is outside all defined phase ranges.", epoch)
            return False

        if new_phase_idx == self._current_phase:
            return False  # No change

        self._current_phase = new_phase_idx
        phase = self._phases[new_phase_idx]
        logger.info(
            "Phase transition -> %s (epoch %d): freeze_backbone=%s, "
            "unfreeze_last_n_blocks=%s, lr_scale=%.4f",
            phase["name"],
            epoch,
            phase["freeze_backbone"],
            phase["unfreeze_last_n_blocks"],
            phase["lr_scale"],
        )
        self._apply_freeze(phase)
        self._rebuild_optimizer(phase)
        return True

    @property
    def current_phase_name(self) -> str | None:
        """Return the name of the currently active phase, or None."""
        if self._current_phase is None:
            return None
        return self._phases[self._current_phase]["name"]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _phase_index_for_epoch(self, epoch: int) -> int | None:
        for i, p in enumerate(self._phases):
            if p["start"] <= epoch <= p["end"]:
                return i
        return None

    def _get_branches(self) -> list[nn.Sequential]:
        """Return the unique branch(es) of BiChannelEncoder."""
        encoder = self.model.encoder  # BiChannelEncoder
        branches = [encoder.left_branch]
        # right_branch may be the same object (shared_weights) — deduplicate
        if hasattr(encoder, "right_branch") and encoder.right_branch is not encoder.left_branch:
            branches.append(encoder.right_branch)
        return branches

    def _apply_freeze(self, phase: dict[str, Any]) -> None:
        """Adjust requires_grad flags for all backbone branches."""
        branches = self._get_branches()
        for branch in branches:
            if phase["freeze_backbone"]:
                # Phase 1: freeze the ENTIRE backbone
                _set_backbone_requires_grad(branch, False)
            else:
                n = phase["unfreeze_last_n_blocks"]
                if n is None:
                    # Phase 3: unfreeze entire backbone
                    _set_backbone_requires_grad(branch, True)
                else:
                    # Phase 2: first freeze all, then selectively unfreeze last n blocks
                    _set_backbone_requires_grad(branch, False)
                    _set_last_n_blocks_requires_grad(branch, n, True)

    def _rebuild_optimizer(self, phase: dict[str, Any]) -> None:
        """
        Rebuild the optimizer parameter groups to reflect the new requires_grad
        state and lr_scale.

        Strategy:
          - Non-backbone params (head, text encoder, fusion, self-attention,
            projection layers) -> base_lr x 1.0  (always trainable)
          - Backbone params that now require grad -> base_lr x lr_scale
        """
        train_cfg = self.cfg.get("training", {})
        base_lr = float(train_cfg.get("base_lr", train_cfg.get("lr", 1e-4)))
        lr_scale = phase["lr_scale"]
        branches = self._get_branches()

        # Collect backbone parameter ids (from unique branches only)
        backbone_param_ids: set[int] = set()
        for branch in branches:
            backbone = branch[0]
            for p in backbone.parameters():
                backbone_param_ids.add(id(p))

        head_params: list[nn.Parameter] = []
        backbone_params: list[nn.Parameter] = []
        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            if id(p) in backbone_param_ids:
                backbone_params.append(p)
            else:
                head_params.append(p)

        param_groups: list[dict[str, Any]] = []
        if head_params:
            param_groups.append({"params": head_params, "lr": base_lr})
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": base_lr * lr_scale})

        # Mutate the existing optimizer in-place: clear and re-add param groups
        # so that any existing scheduler can continue referencing this optimizer.
        weight_decay = float(train_cfg.get("weight_decay", 1e-4))
        betas = tuple(train_cfg.get("betas", (0.9, 0.999)))
        eps = float(train_cfg.get("eps", 1e-8))

        self.optimizer.param_groups.clear()
        for group in param_groups:
            group.setdefault("weight_decay", weight_decay)
            group.setdefault("betas", betas)
            group.setdefault("eps", eps)
            group.setdefault("amsgrad", False)
            self.optimizer.add_param_group(group)

        logger.debug(
            "Optimizer rebuilt: %d head params (lr=%.2e), %d backbone params (lr=%.2e).",
            len(head_params),
            base_lr,
            len(backbone_params),
            base_lr * lr_scale,
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        model: nn.Module,
        cfg: dict[str, Any],
    ) -> "PhaseController":
        """
        Build PhaseController + fresh optimizer from config.

        Returns the controller; the optimizer is accessible via
        ``controller.optimizer``.
        """
        optimizer = build_optimizer(model, cfg)
        return cls(model=model, cfg=cfg, optimizer=optimizer)
