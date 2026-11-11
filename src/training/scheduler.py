"""
src/training/scheduler.py
=========================
P6.2 — Optimizer and learning rate scheduler factory.

Standard configuration:
    - Optimizer: AdamW (weight decay 1e-4, base_lr 1e-4).
    - Scheduler: CosineAnnealingLR (T_max = total_epochs, eta_min = 1e-6).

All parameters are read directly from the YAML config. Supports progressive
unfreezing parameter groups in Phase 6 / Phase Controller (P6.3).
"""

from __future__ import annotations

from typing import Any, Iterable

import torch.nn as nn
from torch.optim import SGD, Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    StepLR,
    _LRScheduler,
)


def build_optimizer(
    model_or_params: nn.Module | Iterable[nn.Parameter] | list[dict[str, Any]],
    cfg: dict[str, Any],
) -> Optimizer:
    """
    Construct optimizer from model and configuration dictionary.

    Args:
        model_or_params: PyTorch module, parameter iterator, or parameter group list.
        cfg:             Configuration dictionary containing a 'training' section.

    Returns:
        Configured PyTorch Optimizer.
    """
    train_cfg = cfg.get("training", {})
    opt_name = str(train_cfg.get("optimizer", "adamw")).lower()
    lr = float(train_cfg.get("base_lr", train_cfg.get("lr", 1e-4)))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    betas = tuple(train_cfg.get("betas", (0.9, 0.999)))
    eps = float(train_cfg.get("eps", 1e-8))

    # Resolve parameters
    if isinstance(model_or_params, nn.Module):
        # Only optimize parameters that require gradients
        params = [p for p in model_or_params.parameters() if p.requires_grad]
    else:
        params = model_or_params

    if opt_name == "adamw":
        return AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
    elif opt_name == "adam":
        return Adam(params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
    elif opt_name == "sgd":
        momentum = float(train_cfg.get("momentum", 0.9))
        return SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(
            f"Unsupported optimizer: '{opt_name}'. Expected 'adamw', 'adam', or 'sgd'."
        )


def build_scheduler(
    optimizer: Optimizer,
    cfg: dict[str, Any],
    total_epochs: int | None = None,
) -> _LRScheduler:
    """
    Construct learning rate scheduler from configuration dictionary.

    Args:
        optimizer:    PyTorch optimizer whose learning rate will be adjusted.
        cfg:          Configuration dictionary containing 'training' section.
        total_epochs: Optional total training epochs override.

    Returns:
        Configured PyTorch _LRScheduler.
    """
    train_cfg = cfg.get("training", {})
    sched_name = str(train_cfg.get("scheduler", "cosine_annealing")).lower()

    epochs = total_epochs or int(train_cfg.get("epochs", 50))
    t_max = int(train_cfg.get("t_max", epochs))
    min_lr = float(train_cfg.get("min_lr", 1e-6))

    if sched_name in ("cosine_annealing", "cosine", "cosineannealinglr"):
        return CosineAnnealingLR(optimizer, T_max=t_max, eta_min=min_lr)
    elif sched_name in ("step", "steplr"):
        step_size = int(train_cfg.get("step_size", 10))
        gamma = float(train_cfg.get("gamma", 0.1))
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif sched_name in ("linear", "linearlr"):
        total_iters = int(train_cfg.get("total_iters", epochs))
        return LinearLR(
            optimizer, start_factor=1.0, end_factor=min_lr / 1e-4, total_iters=total_iters
        )
    else:
        raise ValueError(
            f"Unsupported scheduler: '{sched_name}'. "
            f"Expected 'cosine_annealing', 'step', or 'linear'."
        )


def build_optimizer_and_scheduler(
    model_or_params: nn.Module | Iterable[nn.Parameter] | list[dict[str, Any]],
    cfg: dict[str, Any],
    total_epochs: int | None = None,
) -> tuple[Optimizer, _LRScheduler]:
    """
    Convenience factory building both optimizer and scheduler together.
    """
    optimizer = build_optimizer(model_or_params, cfg)
    scheduler = build_scheduler(optimizer, cfg, total_epochs=total_epochs)
    return optimizer, scheduler
