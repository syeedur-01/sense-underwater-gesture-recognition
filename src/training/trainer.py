"""
src/training/trainer.py
=======================
P6.3 — Three-phase progressive unfreezing controller (PhaseController).
P6.8 — Trainer class: ties together dataloaders, model, loss,
        optimizer/scheduler, PhaseController, checkpointing,
        ExperimentTracker, and early stopping.

Key invariant (P6.8):
  .fit() trains on TRAIN and evaluates on VAL only.
  It never opens or touches any test-split CSV.
  .validate() is usable standalone but its docstring explicitly states it must
  only be called against val during training/tuning; each run's designated
  test split is reserved exclusively for its own evaluation phase.
"""

from __future__ import annotations

import logging
import math
import shutil
import time
from pathlib import Path
from typing import Any

import torch
import torch.amp
import torch.nn as nn
from torch.optim import Optimizer

from src.data.caddy_dataset import get_dataloader
from src.data.transforms import eval_transforms, train_transforms
from src.evaluation.metrics import (
    compute_macro_f1,
    compute_min_class_accuracy,
    compute_overall_accuracy,
    is_better_checkpoint,
)
from src.training.losses import SenseLoss
from src.training.scheduler import build_optimizer, build_scheduler
from src.utils.checkpoint import save_checkpoint
from src.utils.logging import ExperimentTracker
from src.utils.seed import set_seed

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
        scheduler: Any = None,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.optimizer = optimizer
        self.scheduler = scheduler
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
        self._rebuild_optimizer(phase, epoch=epoch)
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

    def _rebuild_optimizer(self, phase: dict[str, Any], epoch: int | None = None) -> None:
        """
        Rebuild the optimizer parameter groups to reflect the new requires_grad
        state and lr_scale, preserving the cosine decay schedule without jumps.

        Strategy:
          - Non-backbone params (head, text encoder, fusion, self-attention,
            projection layers) -> base_lr x 1.0  (always trainable)
          - Backbone params that now require grad -> base_lr x lr_scale
          - Learning rates are scaled by the cosine factor for the current epoch
            so that unfreezing does not cause an abrupt learning rate spike.
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

        # When a scheduler is attached, scale newly rebuilt learning rates by
        # the current epoch's decay factor so unfreezing does not spike LR.
        # When standalone (no scheduler attached), keep base_lr and base_lr * lr_scale.
        if self.scheduler is not None and epoch is not None and epoch > 1:
            t_max = int(train_cfg.get("t_max", train_cfg.get("epochs", 50)))
            if hasattr(self.scheduler, "eta_min_ratio"):
                eta_min_ratio = float(self.scheduler.eta_min_ratio)
            elif "scheduler_eta_min_ratio" in train_cfg:
                eta_min_ratio = float(train_cfg["scheduler_eta_min_ratio"])
            elif "min_lr" in train_cfg and base_lr > 0:
                eta_min_ratio = float(train_cfg["min_lr"]) / base_lr
            else:
                eta_min_ratio = float(train_cfg["scheduler_eta_min_ratio"])
            s = max(0, epoch - 1)
            cos_fac = 0.5 * (1.0 + math.cos(math.pi * s / t_max))
            decay_factor = eta_min_ratio + (1.0 - eta_min_ratio) * cos_fac
        else:
            decay_factor = 1.0

        param_groups: list[dict[str, Any]] = []
        if head_params:
            param_groups.append(
                {
                    "params": head_params,
                    "lr": base_lr * decay_factor,
                    "initial_lr": base_lr,
                }
            )
        if backbone_params:
            param_groups.append(
                {
                    "params": backbone_params,
                    "lr": (base_lr * lr_scale) * decay_factor,
                    "initial_lr": base_lr * lr_scale,
                }
            )

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

        if self.scheduler is not None:
            self.scheduler.base_lrs = [g["initial_lr"] for g in self.optimizer.param_groups]

        logger.debug(
            "Optimizer rebuilt: %d head params (lr=%.2e), %d backbone params (lr=%.2e).",
            len(head_params),
            base_lr * decay_factor,
            len(backbone_params),
            (base_lr * lr_scale) * decay_factor,
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        model: nn.Module,
        cfg: dict[str, Any],
        scheduler: Any = None,
    ) -> "PhaseController":
        """
        Build PhaseController + fresh optimizer from config.

        Returns the controller; the optimizer is accessible via
        ``controller.optimizer``.
        """
        optimizer = build_optimizer(model, cfg)
        return cls(model=model, cfg=cfg, optimizer=optimizer, scheduler=scheduler)


# ---------------------------------------------------------------------------
# P6.8 — Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """
    P6.8 — Full SENSE training loop.

    Ties together:
      - CaddyStereoDataset dataloaders (train + val, never test)
      - SENSE model
      - SenseLoss (combined CE + contrastive, class-weighted)
      - AdamW optimizer + CosineAnnealingLR scheduler
      - PhaseController (progressive unfreezing)
      - ExperimentTracker (nine-artifact experiment record)
      - Checkpoint selection (best = highest val macro F1; tie-break = min-class accuracy)
      - Early stopping (patience from config)
      - Divergence guard: fail-fast abort (RuntimeError) on ``nan_step_tolerance``
        consecutive non-finite training losses or a single non-finite validation
        loss, raised BEFORE any end-of-epoch checkpoint save.  A single occasional
        non-finite step is tolerated and skipped (mixed-precision loss scaling
        legitimately skips the occasional overflow).
      - Gradient diagnostics: mean unscaled gradient norm and clipping-active
        fraction per epoch, logged to training.log only (audit item 4).

    INVARIANT: ``.fit()`` only reads/writes train and val CSVs.
    It **never** opens the test-split CSV or any LOSO test CSV.
    ``validate()`` may be called against any DataLoader, but its docstring
    makes clear that the test split must only be accessed in the dedicated
    evaluation phase (Phase 8 / Phase 9), not during training or tuning.

    Args:
        model:           SENSE model instance (already on the correct device).
        cfg:             Full YAML config dict (merged base + experiment).
        train_csv:       Path to training split CSV (used for dataloading and
                         computing class weights).
        val_csv:         Path to validation split CSV.
        experiment_name: Logical name for this run (used by ExperimentTracker).
        device:          Torch device string (``"cuda"`` or ``"cpu"``).
        resume_path:     Optional checkpoint path to resume from.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: dict[str, Any],
        train_csv: str,
        val_csv: str,
        experiment_name: str = "sense",
        device: str | None = None,
        resume_path: str | None = None,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.train_csv = str(train_csv)
        self.val_csv = str(val_csv)
        self.experiment_name = experiment_name
        self.resume_path = resume_path

        # Device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)

        # Training config
        t_cfg = cfg.get("training", {})
        self.num_epochs: int = int(t_cfg.get("epochs", 50))
        self.batch_size: int = int(t_cfg.get("batch_size", 16))
        self.patience: int = int(t_cfg.get("early_stopping_patience", 10))
        self.seed: int = int(t_cfg.get("seed", 42))

        # Image size from config
        image_size: int = int(cfg.get("data", {}).get("image_size", 224))

        # Enhancement method
        enhancement_method: str | None = cfg.get("data", {}).get("enhancement_method", None)

        # Number of workers / pin memory
        num_workers: int = int(cfg.get("data", {}).get("num_workers", 4))
        pin_memory: bool = bool(cfg.get("data", {}).get("pin_memory", True))

        # Build dataloaders — ONLY train + val
        self.train_loader = get_dataloader(
            df_or_path=self.train_csv,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            transform=train_transforms(image_size),
            enhancement_method=enhancement_method,
            pin_memory=pin_memory,
        )
        self.val_loader = get_dataloader(
            df_or_path=self.val_csv,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            transform=eval_transforms(image_size),
            enhancement_method=enhancement_method,
            pin_memory=pin_memory,
        )

        # Build loss, optimizer, scheduler
        # SenseLoss.from_config() internally computes class weights from train_csv
        self.loss_fn = SenseLoss.from_config(cfg, train_csv=self.train_csv, device=self.device).to(
            self.device
        )
        self.optimizer = build_optimizer(model, cfg)
        self.scheduler = build_scheduler(self.optimizer, cfg)

        # Phase controller (progressive unfreezing)
        self.phase_ctrl = PhaseController(
            model=self.model,
            cfg=cfg,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )

        # Determine the first unfreezing epoch from config training.phases
        phases_cfg = cfg.get("training", {}).get("phases", {})
        first_unfreeze = None
        for p_name in ("phase1", "phase2", "phase3"):
            p_info = phases_cfg.get(p_name, {})
            if not p_info.get("freeze_backbone", True):
                first_unfreeze = int(p_info["epochs"][0])
                break
        if first_unfreeze is None:
            first_unfreeze = int(phases_cfg.get("phase2", {}).get("epochs", [11])[0])
        self._first_unfreeze_epoch = first_unfreeze

        # Experiment tracker
        self.tracker = ExperimentTracker(
            experiment_name=self.experiment_name,
            cfg=self.cfg,
        )

        # Mixed precision scaler
        use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)
        self.use_amp = use_amp

        # State for early stopping / checkpoint selection
        self._best_metrics: dict[str, float] = {}
        self._max_macro_f1_seen: float = 0.0
        self._epochs_no_improve: int = 0
        self._start_epoch: int = 1

        # Divergence guard state (fail-fast on NaN/inf).
        # Internal safeguard, deliberately NOT config-driven.  A single
        # occasional non-finite step is tolerated (mixed-precision loss scaling
        # legitimately skips the occasional overflow); ``nan_step_tolerance``
        # CONSECUTIVE non-finite training losses, or a single non-finite
        # validation loss, means the run has diverged and training aborts.
        self.nan_step_tolerance: int = 20
        self._consecutive_nonfinite_steps: int = 0
        self._nonfinite_warned: bool = False
        # Epoch currently being trained (read by _validate_epoch for guard
        # messages; kept as state so _validate_epoch's signature stays fixed).
        self._current_epoch: int = 0

        # Per-epoch gradient statistics (audit item 4), refreshed each epoch by
        # _train_epoch and logged to training.log (never to metrics.csv).
        self._last_grad_stats: dict[str, float] = {}

        # Resume if requested
        if resume_path is not None:
            self._resume(resume_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, num_epochs: int | None = None) -> dict[str, float]:
        """
        Train for ``num_epochs`` epochs (defaults to ``training.epochs`` from config).

        Evaluates on the validation split at the end of every epoch.
        NEVER accesses any test-split CSV.

        Returns:
            Best validation metrics dict (macro_f1, accuracy, min_class_accuracy).
        """
        set_seed(self.seed)
        if num_epochs is not None:
            end_epoch = self._start_epoch + num_epochs
        else:
            end_epoch = max(self._start_epoch, self.num_epochs + 1)
        self.tracker.start(seed=self.seed)

        try:
            for epoch in range(self._start_epoch, end_epoch):
                epoch_start = time.monotonic()
                self._current_epoch = epoch

                # Phase transition (progressive unfreezing)
                transitioned = self.phase_ctrl.step(epoch)
                if transitioned:
                    logger.info(
                        "Phase transition at epoch %d (%s): resetting early stopping counter.",
                        epoch,
                        self.phase_ctrl.current_phase_name,
                    )
                    self._epochs_no_improve = 0

                # Train one epoch
                train_metrics = self._train_epoch(epoch)

                # Validate (divergence guard inside: raises on a non-finite val
                # loss BEFORE the checkpoint saves below)
                val_metrics = self._validate_epoch()

                # Scheduler step
                self.scheduler.step()
                lr = self.scheduler.get_last_lr()[0]

                epoch_duration = time.monotonic() - epoch_start

                # Log to tracker (CSV + TensorBoard)
                self.tracker.log_epoch(
                    epoch=epoch,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    learning_rate=lr,
                    epoch_duration_s=epoch_duration,
                )

                # Evaluate improvement and save best checkpoint if improved.
                # Never overwrite best.pth with non-finite metrics or weights.
                if self._is_improved(val_metrics):
                    if self._checkpoint_finite(val_metrics):
                        self._best_metrics = dict(val_metrics)
                        self._epochs_no_improve = 0
                        save_checkpoint(
                            path=self.tracker.best_checkpoint_path,
                            model=self.model,
                            epoch=epoch,
                            metrics=val_metrics,
                            config=self.cfg,
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                            best_metrics=self._best_metrics,
                            max_macro_f1_seen=self._max_macro_f1_seen,
                            epochs_no_improve=self._epochs_no_improve,
                            phase_name=self.phase_ctrl.current_phase_name,
                        )
                        logger.info("New best checkpoint: macro_f1=%.4f", val_metrics["macro_f1"])
                    else:
                        logger.error(
                            "Refusing to overwrite best.pth at epoch %d: "
                            "checkpoint metrics or model weights are non-finite.",
                            epoch,
                        )
                        self._epochs_no_improve += 1
                else:
                    self._epochs_no_improve += 1

                # Save last checkpoint (reflects latest end-of-epoch state).
                # Same finiteness gate as best.pth: keep the previous healthy
                # checkpoint rather than persisting a diverged state for resume.
                if self._checkpoint_finite(val_metrics):
                    save_checkpoint(
                        path=self.tracker.last_checkpoint_path,
                        model=self.model,
                        epoch=epoch,
                        metrics=val_metrics,
                        config=self.cfg,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        best_metrics=self._best_metrics,
                        max_macro_f1_seen=self._max_macro_f1_seen,
                        epochs_no_improve=self._epochs_no_improve,
                        phase_name=self.phase_ctrl.current_phase_name,
                    )
                else:
                    logger.error(
                        "Refusing to update last.pth at epoch %d: "
                        "checkpoint metrics or model weights are non-finite; "
                        "the previous checkpoint is kept.",
                        epoch,
                    )

                # Early stopping check
                if epoch >= self._first_unfreeze_epoch and self._epochs_no_improve >= self.patience:
                    logger.info(
                        "Early stopping triggered at epoch %d after %d epochs with no improvement.",
                        epoch,
                        self.patience,
                    )
                    break

                # Save last checkpoint (reflects latest end-of-epoch state).
                # Same finiteness gate as best.pth: keep the previous healthy
                # checkpoint rather than persisting a diverged state for resume.
                if self._checkpoint_finite(val_metrics):
                    save_checkpoint(
                        path=self.tracker.last_checkpoint_path,
                        model=self.model,
                        epoch=epoch,
                        metrics=val_metrics,
                        config=self.cfg,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        best_metrics=self._best_metrics,
                        max_macro_f1_seen=self._max_macro_f1_seen,
                        epochs_no_improve=self._epochs_no_improve,
                        phase_name=self.phase_ctrl.current_phase_name,
                    )
                else:
                    logger.error(
                        "Refusing to update last.pth at epoch %d: "
                        "checkpoint metrics or model weights are non-finite; "
                        "the previous checkpoint is kept.",
                        epoch,
                    )

                # Early stopping check
                if epoch >= self._first_unfreeze_epoch and self._epochs_no_improve >= self.patience:
                    logger.info(
                        "Early stopping triggered at epoch %d after %d epochs with no improvement.",
                        epoch,
                        self.patience,
                    )
                    break

        finally:
            self.tracker.finish()

        return self._best_metrics

    def validate(self, loader: torch.utils.data.DataLoader | None = None) -> dict[str, float]:
        """
        Evaluate the model on ``loader`` (defaults to the val DataLoader).

        WARNING: This method must only be called against the **validation** split
        during training/hyperparameter tuning.  Test splits (standard_test.csv or
        any LOSO fold test CSV) must be reserved exclusively for the final
        evaluation phase (Phase 8 / Phase 9).  Calling this against a test split
        during training constitutes data leakage.

        Returns:
            Dict with keys: loss, accuracy, macro_f1, min_class_accuracy.
        """
        if loader is None:
            loader = self.val_loader
        return self._validate_epoch(loader=loader)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        """Run one training epoch. Returns train metric dict.

        Also records per-epoch gradient diagnostics (audit item 4): the mean
        unscaled gradient norm and the fraction of steps where gradient
        clipping was active, logged to training.log (never to metrics.csv).
        """
        self.model.train()
        total_loss = total_ce = total_contrastive = 0.0
        n_batches = 0
        # Gradient-norm statistics: captured from clip_grad_norm_'s return value
        # (the pre-clip unscaled total norm) — no extra computation required.
        grad_norm_sum = 0.0
        n_grad_steps = 0
        n_clipped = 0
        clip_max_norm = 1.0  # unchanged P6.8 value

        for batch in self.train_loader:
            left = batch["left"].to(self.device, non_blocking=True)
            right = batch["right"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(left, right if self.model.stereo_enabled else None)
                loss_val, components = self.loss_fn(logits, labels, return_components=True)

            # --- Fail-fast divergence guard -----------------------------------
            # A non-finite forward loss means this batch produced garbage whose
            # gradients cannot be trusted (NaN grads would poison optimizer
            # state).  Skip the batch entirely: no backward, no optimizer step.
            # Occasional single overflows are tolerated (mixed-precision loss
            # scaling skips their update anyway); ``nan_step_tolerance``
            # consecutive non-finite losses means the run has diverged.
            loss_f = float(components["loss"])
            if math.isfinite(loss_f):
                self._consecutive_nonfinite_steps = 0
            else:
                self._consecutive_nonfinite_steps += 1
                if not self._nonfinite_warned:
                    self._nonfinite_warned = True
                    logger.warning(
                        "Non-finite training loss (%s) at epoch %d, batch %d "
                        "(consecutive non-finite steps: %d/%d).",
                        loss_f,
                        epoch,
                        n_batches + 1,
                        self._consecutive_nonfinite_steps,
                        self.nan_step_tolerance,
                    )
                if self._consecutive_nonfinite_steps >= self.nan_step_tolerance:
                    logger.error(
                        "Divergence detected: training loss non-finite for %d consecutive "
                        "steps (tolerance %d). Last occurrence: epoch %d, batch %d. "
                        "Aborting before any checkpoint save.",
                        self._consecutive_nonfinite_steps,
                        self.nan_step_tolerance,
                        epoch,
                        n_batches + 1,
                    )
                    raise RuntimeError(
                        f"Training diverged: loss was non-finite for "
                        f"{self._consecutive_nonfinite_steps} consecutive steps "
                        f"(tolerance {self.nan_step_tolerance}) at epoch {epoch}, "
                        f"batch {n_batches + 1}. Last loss value: {loss_f}."
                    )
                # Tolerated single overflow: keep the NaN in the accumulated
                # epoch means (visible in the CSV/TensorBoard record) and skip
                # the poisoned update.
                total_loss += components["loss"]
                total_ce += components["ce_loss"]
                total_contrastive += components["contrastive_loss"]
                n_batches += 1
                continue

            self.scaler.scale(loss_val).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=clip_max_norm
            )
            norm_f = float(grad_norm)
            grad_norm_sum += norm_f
            n_grad_steps += 1
            if norm_f > clip_max_norm:
                n_clipped += 1
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += components["loss"]
            total_ce += components["ce_loss"]
            total_contrastive += components["contrastive_loss"]
            n_batches += 1

        # Per-epoch gradient diagnostics (audit item 4): training.log only,
        # deliberately NOT metrics.csv / TensorBoard.
        n_grad = max(n_grad_steps, 1)
        mean_grad_norm = grad_norm_sum / n_grad
        clip_fraction = n_clipped / n_grad
        skipped_batches = n_batches - n_grad_steps
        self._last_grad_stats = {
            "mean_norm": mean_grad_norm,
            "clip_fraction": clip_fraction,
            "steps": n_grad_steps,
            "clipped": n_clipped,
            "skipped_batches": skipped_batches,
        }
        logger.info(
            "Epoch %d gradient stats — mean unscaled grad norm: %.4f; "
            "clipping active in %.1f%% of steps (%d/%d; %d non-finite batches skipped).",
            epoch,
            mean_grad_norm,
            100.0 * clip_fraction,
            n_clipped,
            n_grad_steps,
            skipped_batches,
        )

        n = max(n_batches, 1)
        return {
            "loss": total_loss / n,
            "ce_loss": total_ce / n,
            "contrastive_loss": total_contrastive / n,
        }

    def _validate_epoch(
        self, loader: torch.utils.data.DataLoader | None = None
    ) -> dict[str, float]:
        """Run one validation pass. Returns val metric dict.

        Fail-fast divergence guard: raises RuntimeError when the validation
        loss is non-finite.  In ``fit()`` this fires BEFORE the end-of-epoch
        checkpoint saves, so ``last.pth`` always holds the last healthy state.
        The failing epoch number is read from ``self._current_epoch`` (set by
        ``fit()``) so this method's signature stays unchanged.
        """
        epoch: int | None = getattr(self, "_current_epoch", None)
        if loader is None:
            loader = self.val_loader
        self.model.eval()
        all_preds: list[int] = []
        all_labels: list[int] = []
        total_loss = 0.0
        n_batches = 0
        nonfinite_batches = 0

        with torch.no_grad():
            for batch in loader:
                left = batch["left"].to(self.device, non_blocking=True)
                right = batch["right"].to(self.device, non_blocking=True)
                labels = batch["label"].to(self.device, non_blocking=True)

                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    logits = self.model(left, right if self.model.stereo_enabled else None)
                    loss_val = self.loss_fn(logits, labels)

                loss_f = float(loss_val.item())
                total_loss += loss_f
                if not math.isfinite(loss_f):
                    nonfinite_batches += 1
                preds = logits.argmax(dim=1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().tolist())
                n_batches += 1

        n = max(n_batches, 1)
        epoch_loss = total_loss / n
        if not math.isfinite(epoch_loss):
            epoch_label = epoch if epoch is not None else "<unknown>"
            logger.error(
                "Divergence detected: validation loss is non-finite at epoch %s "
                "(%d/%d val batches non-finite). Aborting before any checkpoint save.",
                epoch_label,
                nonfinite_batches,
                n_batches,
            )
            raise RuntimeError(
                f"Validation loss is non-finite at epoch {epoch_label} "
                f"({nonfinite_batches}/{n_batches} val batches non-finite); the model "
                f"has likely diverged. No checkpoint was saved for this epoch."
            )
        return {
            "loss": epoch_loss,
            "accuracy": compute_overall_accuracy(all_labels, all_preds),
            "macro_f1": compute_macro_f1(all_labels, all_preds),
            "min_class_accuracy": compute_min_class_accuracy(all_labels, all_preds),
        }

    def _is_improved(self, val_metrics: dict[str, float]) -> bool:
        """
        Return True if val_metrics is better than the current best.

        The 0.005 tie-breaking window compares the candidate's macro F1 against
        ``_max_macro_f1_seen`` (the highest macro F1 seen across all evaluated epochs),
        NOT against the last accepted checkpoint, preventing creeping degradation.
        """
        current_f1 = float(val_metrics.get("macro_f1", 0.0))
        current_min_acc = float(val_metrics.get("min_class_accuracy", 0.0))

        if not self._best_metrics:
            self._max_macro_f1_seen = current_f1
            return True

        improved = is_better_checkpoint(
            val_macro_f1=current_f1,
            val_min_acc=current_min_acc,
            best_macro_f1=self._max_macro_f1_seen,
            best_min_acc=(
                float(self._best_metrics["min_class_accuracy"])
                if "min_class_accuracy" in self._best_metrics
                else None
            ),
        )
        if current_f1 > self._max_macro_f1_seen:
            self._max_macro_f1_seen = current_f1
        return improved

    def _metrics_finite(self, metrics: dict[str, float]) -> bool:
        """True when every metric value is a finite number (no NaN/inf)."""
        return all(
            isinstance(v, (int, float)) and math.isfinite(float(v)) for v in metrics.values()
        )

    def _model_state_finite(self) -> bool:
        """True when every floating-point parameter and buffer of the model is finite."""
        tensors = list(self.model.parameters()) + list(self.model.buffers())
        for t in tensors:
            if t.is_floating_point() and not bool(torch.isfinite(t).all()):
                return False
        return True

    def _checkpoint_finite(self, metrics: dict[str, float]) -> bool:
        """
        Gate applied before overwriting ``best.pth`` or ``last.pth``:
        both the metrics dict and the model state must be finite.
        """
        return self._metrics_finite(metrics) and self._model_state_finite()

    def _resume(self, path: str) -> None:
        """
        Load a checkpoint to resume training from.

        Restores:
          1. Model weights
          2. Phase state for the resume epoch: backbone requires_grad flags and
             optimizer parameter groups rebuilt BEFORE loading optimizer state.
          3. Optimizer and scheduler state dicts.
          4. _best_metrics and _max_macro_f1_seen (preventing degradation / overwriting best.pth).
          5. _epochs_no_improve (early-stopping counter).
          6. best.pth preserved in the destination checkpoint directory if present.
        """
        from src.utils.checkpoint import load_checkpoint

        payload = load_checkpoint(
            path,
            model=self.model,
            optimizer=None,
            scheduler=None,
            map_location=str(self.device),
        )
        saved_epoch = int(payload.get("epoch", 1))
        self._start_epoch = saved_epoch + 1

        # Restore phase controller state and rebuild optimizer groups for this phase
        # BEFORE loading optimizer/scheduler states:
        phase_idx = self.phase_ctrl._phase_index_for_epoch(saved_epoch)
        if phase_idx is not None:
            self.phase_ctrl._current_phase = phase_idx
            phase = self.phase_ctrl._phases[phase_idx]
            self.phase_ctrl._apply_freeze(phase)
            self.phase_ctrl._rebuild_optimizer(phase, epoch=saved_epoch)

        # Restore optimizer and scheduler state dicts
        if "optimizer" in payload and self.optimizer is not None:
            self.optimizer.load_state_dict(payload["optimizer"])
        if "scheduler" in payload and self.scheduler is not None:
            self.scheduler.load_state_dict(payload["scheduler"])

        # Restore metrics and tracking state
        self._best_metrics = dict(payload.get("best_metrics", payload.get("metrics", {})))
        self._max_macro_f1_seen = float(
            payload.get("max_macro_f1_seen", self._best_metrics.get("macro_f1", 0.0))
        )
        self._epochs_no_improve = int(payload.get("epochs_no_improve", 0))

        # Preserve best checkpoint if it exists in the source checkpoint directory
        source_best = Path(path).parent / "best.pth"
        if (
            source_best.exists()
            and self.tracker.best_checkpoint_path.resolve() != source_best.resolve()
        ):
            self.tracker.best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_best, self.tracker.best_checkpoint_path)

        logger.info(
            "Resumed from checkpoint %s at epoch %d (phase=%s, best_macro_f1=%.4f, epochs_no_improve=%d).",
            path,
            saved_epoch,
            self.phase_ctrl.current_phase_name,
            self._max_macro_f1_seen,
            self._epochs_no_improve,
        )
