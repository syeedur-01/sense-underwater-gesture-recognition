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
import time
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
        self.phase_ctrl = PhaseController(model=self.model, cfg=cfg, optimizer=self.optimizer)

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
        self._epochs_no_improve: int = 0
        self._start_epoch: int = 1

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
        num_epochs = num_epochs if num_epochs is not None else self.num_epochs
        self.tracker.start(seed=self.seed)

        try:
            for epoch in range(self._start_epoch, self._start_epoch + num_epochs):
                epoch_start = time.monotonic()

                # Phase transition (progressive unfreezing)
                self.phase_ctrl.step(epoch)

                # Train one epoch
                train_metrics = self._train_epoch(epoch)

                # Validate
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

                # Save last checkpoint
                save_checkpoint(
                    path=self.tracker.last_checkpoint_path,
                    model=self.model,
                    epoch=epoch,
                    metrics=val_metrics,
                    config=self.cfg,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                )

                # Save best checkpoint if improved
                if self._is_improved(val_metrics):
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
                    )
                    logger.info("New best checkpoint: macro_f1=%.4f", val_metrics["macro_f1"])
                else:
                    self._epochs_no_improve += 1
                    if self._epochs_no_improve >= self.patience:
                        logger.info(
                            "Early stopping triggered after %d epochs with no improvement.",
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
        """Run one training epoch. Returns train metric dict."""
        self.model.train()
        total_loss = total_ce = total_contrastive = 0.0
        n_batches = 0

        for batch in self.train_loader:
            left = batch["left"].to(self.device, non_blocking=True)
            right = batch["right"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(left, right if self.model.stereo_enabled else None)
                loss_val, components = self.loss_fn(logits, labels, return_components=True)

            self.scaler.scale(loss_val).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += components["loss"]
            total_ce += components["ce_loss"]
            total_contrastive += components["contrastive_loss"]
            n_batches += 1

        n = max(n_batches, 1)
        return {
            "loss": total_loss / n,
            "ce_loss": total_ce / n,
            "contrastive_loss": total_contrastive / n,
        }

    def _validate_epoch(
        self, loader: torch.utils.data.DataLoader | None = None
    ) -> dict[str, float]:
        """Run one validation pass. Returns val metric dict."""
        if loader is None:
            loader = self.val_loader
        self.model.eval()
        all_preds: list[int] = []
        all_labels: list[int] = []
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in loader:
                left = batch["left"].to(self.device, non_blocking=True)
                right = batch["right"].to(self.device, non_blocking=True)
                labels = batch["label"].to(self.device, non_blocking=True)

                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    logits = self.model(left, right if self.model.stereo_enabled else None)
                    loss_val = self.loss_fn(logits, labels)

                total_loss += loss_val.item()
                preds = logits.argmax(dim=1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().tolist())
                n_batches += 1

        n = max(n_batches, 1)
        return {
            "loss": total_loss / n,
            "accuracy": compute_overall_accuracy(all_labels, all_preds),
            "macro_f1": compute_macro_f1(all_labels, all_preds),
            "min_class_accuracy": compute_min_class_accuracy(all_labels, all_preds),
        }

    def _is_improved(self, val_metrics: dict[str, float]) -> bool:
        """Return True if val_metrics is better than the current best."""
        if not self._best_metrics:
            return True
        return is_better_checkpoint(val_metrics, self._best_metrics)

    def _resume(self, path: str) -> None:
        """Load a checkpoint to resume training from."""
        from src.utils.checkpoint import load_checkpoint

        payload = load_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            map_location=str(self.device),
        )
        self._start_epoch = int(payload.get("epoch", 1)) + 1
        self._best_metrics = dict(payload.get("metrics", {}))
        logger.info("Resumed from checkpoint %s at epoch %d.", path, self._start_epoch - 1)
