"""
src/evaluation/metrics.py
=========================
P8.1 — Core metrics module for SENSE diver gesture recognition on CADDY.
(Implemented early as a Phase 6 prerequisite.)

Binding metric definitions according to the Evaluation Strategy:
  - Primary metric: Macro F1-score (penalizes majority-class bias).
  - Secondary metric: Overall (micro) accuracy (direct literature comparability).
  - Supporting metrics:
      * Per-class precision, recall, and F1.
      * Per-class accuracy and Minimum Class Accuracy (safety-critical).
      * Confusion matrix + heatmap visualization helper.
      * Inference latency (ms / sample).
      * Tie-breaking checkpoint selection logic (0.5 macro-F1 point window).

All metrics in Phase 6 onward import from this module. Nothing else may define
its own macro F1, accuracy, or minimum class accuracy.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for headless environments / tests
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def _to_numpy_labels(
    y_true: Any,
    y_pred: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Standardize ground-truth and predicted labels into 1D integer NumPy arrays.

    Args:
        y_true: Ground-truth class labels (1D array-like or tensor of shape (N,)).
        y_pred: Predictions, either class indices (shape (N,)) or 2D logits/probs
                (shape (N, num_classes)).

    Returns:
        (y_true_np, y_pred_np): Both 1D integer NumPy arrays of shape (N,).
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true_np = np.asarray(y_true)
    y_pred_np = np.asarray(y_pred)

    # Flatten if 2D with single column
    if y_true_np.ndim == 2 and y_true_np.shape[1] == 1:
        y_true_np = y_true_np.ravel()

    # If y_pred contains 2D logits/probs (N, C), take argmax across class dimension
    if y_pred_np.ndim == 2:
        if y_pred_np.shape[1] > 1:
            y_pred_np = np.argmax(y_pred_np, axis=1)
        else:
            y_pred_np = y_pred_np.ravel()

    y_true_np = y_true_np.astype(int)
    y_pred_np = y_pred_np.astype(int)

    if len(y_true_np) != len(y_pred_np):
        raise ValueError(
            f"Length mismatch: len(y_true)={len(y_true_np)} vs len(y_pred)={len(y_pred_np)}"
        )

    return y_true_np, y_pred_np


def compute_overall_accuracy(y_true: Any, y_pred: Any) -> float:
    """
    Compute overall (micro) accuracy.

    Args:
        y_true: Ground-truth class labels.
        y_pred: Predicted class labels or logits.

    Returns:
        Overall accuracy as a float in [0.0, 1.0]. Returns 0.0 if empty.
    """
    y_true_np, y_pred_np = _to_numpy_labels(y_true, y_pred)
    if len(y_true_np) == 0:
        return 0.0
    return float(accuracy_score(y_true_np, y_pred_np))


def compute_macro_f1(
    y_true: Any,
    y_pred: Any,
    num_classes: int | None = None,
) -> float:
    """
    Compute Macro F1-score (unweighted mean of F1 scores across classes).

    This is the primary metric per the Evaluation Strategy.
    If num_classes is provided, all classes in range(num_classes) are explicitly
    evaluated, ensuring absent classes are penalized rather than ignored.

    Args:
        y_true: Ground-truth class labels.
        y_pred: Predicted class labels or logits.
        num_classes: Optional total number of classes (from config).

    Returns:
        Macro F1-score as a float in [0.0, 1.0]. Returns 0.0 if empty.
    """
    y_true_np, y_pred_np = _to_numpy_labels(y_true, y_pred)
    if len(y_true_np) == 0:
        return 0.0

    labels = list(range(num_classes)) if num_classes is not None else None
    return float(f1_score(y_true_np, y_pred_np, average="macro", labels=labels, zero_division=0.0))


def compute_per_class_accuracy(
    y_true: Any,
    y_pred: Any,
    num_classes: int | None = None,
) -> dict[int, float]:
    """
    Compute per-class accuracy (recall per class: TP_c / Total_c).

    Args:
        y_true: Ground-truth class labels.
        y_pred: Predicted class labels or logits.
        num_classes: Optional total number of classes (from config).

    Returns:
        Dictionary mapping class index (int) -> accuracy (float in [0.0, 1.0]).
        Classes with zero ground-truth samples receive 0.0.
    """
    y_true_np, y_pred_np = _to_numpy_labels(y_true, y_pred)

    if num_classes is not None:
        classes = list(range(num_classes))
    else:
        if len(y_true_np) == 0:
            return {}
        classes = sorted(list(set(y_true_np.tolist()) | set(y_pred_np.tolist())))

    per_class: dict[int, float] = {}
    for c in classes:
        mask = y_true_np == c
        total = int(np.sum(mask))
        if total > 0:
            correct = int(np.sum((y_pred_np == c) & mask))
            per_class[c] = float(correct / total)
        else:
            per_class[c] = 0.0

    return per_class


def compute_min_class_accuracy(
    y_true: Any,
    y_pred: Any,
    num_classes: int | None = None,
    only_present: bool = True,
) -> float:
    """
    Compute Minimum Class Accuracy across classes.

    Safety-relevant metric retained explicitly for comparability with DARE (92%)
    and used as the tie-breaker for checkpoint selection.

    Args:
        y_true: Ground-truth class labels.
        y_pred: Predicted class labels or logits.
        num_classes: Optional total number of classes (from config).
        only_present: If True (default), minimum is computed only across classes
                      that have at least one ground-truth sample present in y_true.
                      If False, classes with 0 samples are included with 0.0 accuracy.

    Returns:
        Minimum class accuracy as a float in [0.0, 1.0].
    """
    y_true_np, y_pred_np = _to_numpy_labels(y_true, y_pred)
    if len(y_true_np) == 0:
        return 0.0

    if num_classes is not None:
        classes = list(range(num_classes))
    else:
        classes = sorted(list(set(y_true_np.tolist())))

    present_accuracies: list[float] = []
    for c in classes:
        mask = y_true_np == c
        total = int(np.sum(mask))
        if total > 0:
            correct = int(np.sum((y_pred_np == c) & mask))
            present_accuracies.append(float(correct / total))
        elif not only_present:
            present_accuracies.append(0.0)

    if not present_accuracies:
        return 0.0

    return float(min(present_accuracies))


def compute_precision_recall_f1_per_class(
    y_true: Any,
    y_pred: Any,
    num_classes: int | None = None,
) -> dict[int, dict[str, float]]:
    """
    Compute precision, recall, F1-score, and support for each class.

    Args:
        y_true: Ground-truth class labels.
        y_pred: Predicted class labels or logits.
        num_classes: Optional total number of classes (from config).

    Returns:
        Dictionary mapping class_id -> {
            "precision": float,
            "recall": float,
            "f1": float,
            "support": int,
        }.
    """
    y_true_np, y_pred_np = _to_numpy_labels(y_true, y_pred)
    labels = list(range(num_classes)) if num_classes is not None else None

    prec, rec, f1, supp = precision_recall_fscore_support(
        y_true_np,
        y_pred_np,
        labels=labels,
        zero_division=0.0,
    )

    actual_labels = (
        labels
        if labels is not None
        else sorted(list(set(y_true_np.tolist()) | set(y_pred_np.tolist())))
    )
    if len(actual_labels) != len(prec):
        actual_labels = list(range(len(prec)))

    res: dict[int, dict[str, float]] = {}
    for i, c in enumerate(actual_labels):
        res[int(c)] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(supp[i]),
        }
    return res


def compute_confusion_matrix(
    y_true: Any,
    y_pred: Any,
    num_classes: int | None = None,
    normalize: str | None = None,
) -> np.ndarray:
    """
    Compute the confusion matrix.

    Args:
        y_true: Ground-truth class labels.
        y_pred: Predicted class labels or logits.
        num_classes: Optional total number of classes (from config).
        normalize: 'true', 'pred', 'all', or None (default count).

    Returns:
        2D NumPy array of shape (num_classes, num_classes).
    """
    y_true_np, y_pred_np = _to_numpy_labels(y_true, y_pred)
    labels = list(range(num_classes)) if num_classes is not None else None
    return confusion_matrix(y_true_np, y_pred_np, labels=labels, normalize=normalize)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str] | None = None,
    save_path: str | Path | None = None,
    title: str = "Confusion Matrix",
    normalize: bool = False,
    cmap: str = "Blues",
) -> plt.Figure:
    """
    Plot confusion matrix as a heatmap.

    Args:
        cm: Confusion matrix array of shape (C, C).
        class_names: Optional list of class names for axes labels.
        save_path: Optional path to save the generated plot PNG.
        title: Plot title.
        normalize: If True, normalizes matrix rows to sum to 1.0.
        cmap: Matplotlib colormap name.

    Returns:
        matplotlib.figure.Figure object.
    """
    mat = cm.astype(float)
    if normalize:
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        mat = mat / row_sums

    fig, ax = plt.subplots(figsize=(10, 8))

    fmt = ".2f" if normalize else ".0f"
    num_classes = cm.shape[0]

    ticks = (
        class_names
        if (class_names and len(class_names) == num_classes)
        else [str(i) for i in range(num_classes)]
    )

    sns.heatmap(
        mat,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=ticks,
        yticklabels=ticks,
        cbar=True,
        ax=ax,
    )

    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches="tight")

    return fig


def is_better_checkpoint(
    val_macro_f1: float,
    val_min_acc: float,
    best_macro_f1: float | None,
    best_min_acc: float | None,
    f1_tolerance: float = 0.005,
) -> bool:
    """
    Determine if the current checkpoint is better than the historical best,
    following the Evaluation Strategy's stated tie-breaking rule:

    - Primary criterion: Validation Macro F1.
    - Tie-breaking criterion: If within 0.5 macro-F1 points (0.005 on [0, 1] scale,
      or 0.5 on [0, 100] scale), the tie is broken by minimum class accuracy
      (favoring the model that fails least badly on its worst class).
    - If minimum class accuracy is also tied within tolerance, the higher macro F1 wins.

    Args:
        val_macro_f1: Current validation macro F1.
        val_min_acc: Current validation minimum class accuracy.
        best_macro_f1: Best validation macro F1 recorded so far (or None).
        best_min_acc: Best validation minimum class accuracy recorded so far (or None).
        f1_tolerance: Window within which macro F1 is considered a tie. Defaults
                      to 0.005 (0.5 percentage points). Automatically scaled to 0.5
                      if macro F1 is given in 0..100 percentage format.

    Returns:
        bool: True if current checkpoint is better, False otherwise.
    """
    if best_macro_f1 is None or best_min_acc is None:
        return True

    # Adjust tolerance if metrics are formatted as percentages (0..100) instead of (0..1)
    tol = f1_tolerance
    if (val_macro_f1 > 1.0 or best_macro_f1 > 1.0) and tol < 0.1:
        tol = 0.5

    delta_f1 = val_macro_f1 - best_macro_f1

    # Clearly better by more than tolerance window
    if delta_f1 > tol:
        return True

    # Clearly worse by more than tolerance window
    if delta_f1 < -tol:
        return False

    # Within tie-breaker window: higher min class accuracy wins
    if val_min_acc > best_min_acc:
        return True
    if val_min_acc < best_min_acc:
        return False

    # Exact tie on min class accuracy as well: higher macro F1 wins
    return val_macro_f1 > best_macro_f1


def measure_inference_time(
    model: nn.Module,
    dataloader: Any,
    device: torch.device | str,
    num_batches: int = 50,
    warmup_batches: int = 5,
) -> float:
    """
    Measure average inference latency in milliseconds per sample.

    Args:
        model: PyTorch model.
        dataloader: DataLoader yielding batches. Supports dictionary batches
                    ({'left': ..., 'right': ...}), tuples, or single tensors.
        device: Device to run inference on ('cuda' or 'cpu').
        num_batches: Maximum number of timed batches.
        warmup_batches: Initial warmup batches to discard.

    Returns:
        mean latency in ms/sample (float).
    """
    dev = torch.device(device)
    orig_mode = model.training
    model.eval()
    model.to(dev)

    is_cuda = dev.type == "cuda" and torch.cuda.is_available()

    def _extract_inputs(batch: Any) -> tuple[torch.Tensor, torch.Tensor | None, int]:
        if isinstance(batch, dict):
            left = batch["left"].to(dev)
            right = batch.get("right")
            right = right.to(dev) if right is not None else None
            batch_size = left.shape[0]
            return left, right, batch_size
        elif isinstance(batch, (list, tuple)):
            if (
                len(batch) >= 2
                and isinstance(batch[0], torch.Tensor)
                and isinstance(batch[1], torch.Tensor)
            ):
                # Check if batch[1] is right image (same shape) or label (1D)
                if batch[0].shape == batch[1].shape:
                    left = batch[0].to(dev)
                    right = batch[1].to(dev)
                else:
                    left = batch[0].to(dev)
                    right = None
                batch_size = left.shape[0]
                return left, right, batch_size
            else:
                left = batch[0].to(dev)
                batch_size = left.shape[0]
                return left, None, batch_size
        elif isinstance(batch, torch.Tensor):
            left = batch.to(dev)
            return left, None, left.shape[0]
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

    try:
        # Warmup
        warmup_count = 0
        timed_count = 0
        total_time_sec = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                left, right, bs = _extract_inputs(batch)

                if warmup_count < warmup_batches:
                    if right is not None:
                        _ = model(left, right)
                    else:
                        _ = model(left)
                    if is_cuda:
                        torch.cuda.synchronize()
                    warmup_count += 1
                    continue

                if timed_count >= num_batches:
                    break

                if is_cuda:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

                if right is not None:
                    _ = model(left, right)
                else:
                    _ = model(left)

                if is_cuda:
                    torch.cuda.synchronize()
                t1 = time.perf_counter()

                total_time_sec += t1 - t0
                total_samples += bs
                timed_count += 1

        if total_samples == 0:
            return 0.0

        ms_per_sample = (total_time_sec / total_samples) * 1000.0
        return float(ms_per_sample)

    finally:
        model.train(orig_mode)


def evaluate_predictions(
    y_true: Any,
    y_pred: Any,
    num_classes: int | None = None,
) -> dict[str, Any]:
    """
    Compute complete evaluation bundle across all defined metrics.

    Args:
        y_true: Ground-truth class labels.
        y_pred: Predicted class labels or logits.
        num_classes: Optional total number of classes (from config).

    Returns:
        dict containing:
            - 'overall_accuracy': float
            - 'macro_f1': float
            - 'min_class_accuracy': float
            - 'per_class_accuracy': dict[int, float]
            - 'per_class_metrics': dict[int, dict[str, float]]
            - 'confusion_matrix': np.ndarray
    """
    y_true_np, y_pred_np = _to_numpy_labels(y_true, y_pred)

    overall_acc = compute_overall_accuracy(y_true_np, y_pred_np)
    macro_f1 = compute_macro_f1(y_true_np, y_pred_np, num_classes=num_classes)
    per_class_acc = compute_per_class_accuracy(y_true_np, y_pred_np, num_classes=num_classes)
    min_acc = compute_min_class_accuracy(
        y_true_np, y_pred_np, num_classes=num_classes, only_present=True
    )
    per_class_metrics = compute_precision_recall_f1_per_class(
        y_true_np, y_pred_np, num_classes=num_classes
    )
    cm = compute_confusion_matrix(y_true_np, y_pred_np, num_classes=num_classes)

    return {
        "overall_accuracy": overall_acc,
        "macro_f1": macro_f1,
        "min_class_accuracy": min_acc,
        "per_class_accuracy": per_class_acc,
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": cm,
    }
