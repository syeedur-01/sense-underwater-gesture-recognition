"""
src/utils/checkpoint.py
========================
P6.4 — Checkpointing utility.

Saves and loads full training state: model weights, optimizer state,
scheduler state, epoch, evaluation metrics (macro F1, accuracy, min-class
accuracy), and the exact YAML config used for the run.

Verification: Bit-identical weights after a save/load round-trip.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

logger = logging.getLogger(__name__)

# Keys that must be present in a valid checkpoint
_REQUIRED_KEYS = {"model", "epoch", "metrics", "config"}


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    epoch: int,
    metrics: dict[str, float],
    config: dict[str, Any],
    optimizer: Optimizer | None = None,
    scheduler: _LRScheduler | None = None,
) -> None:
    """
    Persist a training checkpoint to ``path``.

    The checkpoint contains:
      - ``model``      : model state dict
      - ``optimizer``  : optimizer state dict (omitted when None)
      - ``scheduler``  : scheduler state dict (omitted when None)
      - ``epoch``      : current epoch (1-indexed)
      - ``metrics``    : dict with at least ``macro_f1``, ``accuracy``,
                         ``min_class_accuracy``
      - ``config``     : deep copy of the full YAML config dict

    Args:
        path:      Destination file path (parent directories are created).
        model:     The model whose weights are saved.
        epoch:     Current epoch number (1-indexed).
        metrics:   Dict of metric names -> values to persist.
        config:    Full YAML config dict for this run.
        optimizer: Optional optimizer; its state is included when provided.
        scheduler: Optional LR scheduler; its state is included when provided.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "epoch": epoch,
        "metrics": dict(metrics),
        "config": config,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()

    torch.save(payload, path)
    logger.info(
        "Checkpoint saved → %s  (epoch=%d, macro_f1=%.4f)",
        path,
        epoch,
        float(metrics.get("macro_f1", float("nan"))),
    )


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_checkpoint(
    path: str | Path,
    model: nn.Module | None = None,
    optimizer: Optimizer | None = None,
    scheduler: _LRScheduler | None = None,
    map_location: str | torch.device | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """
    Load a checkpoint from ``path``.

    Optionally restores state into ``model``, ``optimizer``, and ``scheduler``
    in-place.  When ``model`` is None only the raw payload dict is returned
    (useful for inspecting checkpoints or evaluating on a fresh model instance).

    Args:
        path:         Checkpoint file path.
        model:        If provided, model weights are restored in-place.
        optimizer:    If provided, optimizer state is restored in-place.
        scheduler:    If provided, scheduler state is restored in-place.
        map_location: Passed to ``torch.load`` (e.g. ``'cpu'`` for CPU-only load).
        strict:       Passed to ``model.load_state_dict`` (default True).

    Returns:
        The full checkpoint dict with keys: ``model``, ``epoch``, ``metrics``,
        ``config``, and optionally ``optimizer`` / ``scheduler``.

    Raises:
        FileNotFoundError: When ``path`` does not exist.
        KeyError: When a required key is missing from the checkpoint.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    payload: dict[str, Any] = torch.load(path, map_location=map_location, weights_only=False)

    missing = _REQUIRED_KEYS - set(payload.keys())
    if missing:
        raise KeyError(f"Checkpoint at '{path}' is missing required keys: {missing}")

    if model is not None:
        model.load_state_dict(payload["model"], strict=strict)
        logger.info(
            "Model weights restored from %s (epoch=%d).",
            path,
            payload["epoch"],
        )

    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
        logger.info("Optimizer state restored from %s.", path)
    elif optimizer is not None:
        logger.warning("No optimizer state found in checkpoint %s; optimizer not restored.", path)

    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
        logger.info("Scheduler state restored from %s.", path)
    elif scheduler is not None:
        logger.warning("No scheduler state found in checkpoint %s; scheduler not restored.", path)

    return payload
