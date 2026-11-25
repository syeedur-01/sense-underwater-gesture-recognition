"""
scripts/evaluate.py
===================
P8.2 - Generic evaluation entry point for a frozen checkpoint on a held-out split.

Usage (from the project root, venv active):

    python scripts/evaluate.py \
        --checkpoint checkpoints/sense_final_standard_split.pth \
        --split data/processed/splits/standard_test.csv \
        --output results/experiment_1_standard

Design notes
------------
* **No ``--config`` flag.** Every checkpoint written by this project embeds the
  merged config that produced it (``src/utils/checkpoint.py``), and
  ``src.models.factory.build_model`` dispatches on ``model.architecture``. The
  architecture is therefore rebuilt from the checkpoint itself, which removes any
  chance of evaluating a model against a config that did not train it.

* **Test splits only.** P8.2 requires that ``--split`` resolve to a designated
  test partition - ``standard_test.csv``, a ``scenario_holdout_{i}_test.csv``, or
  a scenario-grouped subset of ``standard_test.csv`` - and never a validation
  file. This is enforced, not documented: a ``*_val.csv`` path is rejected.

* **Full precision.** The training loop runs under mixed precision; this script
  runs in float32. The difference between the two paths is small but real, so
  the report states which one produced the number. Both arms are evaluated by
  this same code path, so comparisons between them are unaffected.

* **No interpretation.** This script measures. It does not decide whether a
  difference is meaningful; that is recorded in the knowledge base.

Outputs (all inside ``--output``)
---------------------------------
    metrics.json              complete metric bundle plus provenance
    per_class.csv             one row per class: support, accuracy, P/R/F1
    confusion_matrix.npy      raw counts, for plotting
    confusion_matrix.png      counts heat map
    confusion_matrix_norm.png row-normalised heat map
    report.md                 human-readable summary

Invariant: this script only ever reads the split it is given. It never writes to
``runs/`` and never modifies a checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# -- project root on sys.path so `src` imports work from any CWD ------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.data.caddy_dataset import get_dataloader  # noqa: E402
from src.data.transforms import eval_transforms  # noqa: E402
from src.evaluation.metrics import evaluate_predictions, plot_confusion_matrix  # noqa: E402
from src.models.factory import build_model  # noqa: E402
from src.models.text_encoder import CADDY_CLASS_NAMES  # noqa: E402
from src.utils.checkpoint import load_checkpoint  # noqa: E402

__all__ = ["main", "run_inference", "sha256_file"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """Return the SHA-256 of a file, upper-case hex, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def resolve_device(requested: str | None) -> torch.device:
    """Resolve the evaluation device, defaulting to CUDA when available."""
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _jsonable(value):
    """Recursively convert numpy/pandas types into JSON-serialisable objects."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def run_inference(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    """
    Run one deterministic inference pass and return (y_true, y_pred).

    The forward call mirrors the trainer's validation path exactly: the right
    view is passed only when the model declares ``stereo_enabled``.
    """
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for batch in loader:
            left = batch["left"].to(device, non_blocking=True)
            right = batch["right"].to(device, non_blocking=True)
            labels = batch["label"]

            logits = model(left, right if model.stereo_enabled else None)

            y_true.extend(int(v) for v in labels.tolist())
            y_pred.extend(int(v) for v in logits.argmax(dim=1).cpu().tolist())

    return y_true, y_pred


def reject_validation_split(split_path: Path) -> None:
    """P8.2 guard: refuse any validation CSV."""
    name = split_path.name.lower()
    if "_val" in name or "validation" in name:
        raise SystemExit(
            f"Refusing to evaluate on '{split_path.name}': P8.2 restricts this script to "
            "designated test splits. Validation performance is produced by the training "
            "loop, not by the evaluation script."
        )


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------


def write_per_class_csv(
    out_path: Path,
    per_class_metrics: dict[int, dict[str, float]],
    confusion: np.ndarray,
    class_names: list[str],
) -> None:
    """Write one row per class: support, correct, accuracy, precision, recall, F1."""
    supports = confusion.sum(axis=1)
    corrects = np.diag(confusion)

    rows = []
    for class_id in sorted(per_class_metrics):
        metrics = per_class_metrics[class_id]
        support = int(supports[class_id])
        correct = int(corrects[class_id])
        rows.append(
            {
                "class_id": class_id,
                "class_name": (
                    class_names[class_id] if class_id < len(class_names) else str(class_id)
                ),
                "support": support,
                "correct": correct,
                "accuracy": (correct / support) if support else float("nan"),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            }
        )

    pd.DataFrame(rows).to_csv(out_path, index=False, float_format="%.6f")


def write_report(
    out_path: Path,
    label: str,
    provenance: dict,
    metrics: dict,
    per_class_metrics: dict[int, dict[str, float]],
    class_names: list[str],
) -> None:
    """Write the Markdown summary."""
    confusion = np.asarray(metrics["confusion_matrix"])
    supports = confusion.sum(axis=1)

    lines: list[str] = []
    lines.append(f"# Evaluation report - {label}")
    lines.append("")
    lines.append(
        "Generated by the shared evaluation entry point. Every value below is a "
        "test-partition measurement produced in a single inference pass."
    )
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|---|---|")
    lines.append(f"| Checkpoint | `{provenance['checkpoint_path']}` |")
    lines.append(f"| Checkpoint SHA-256 | `{provenance['checkpoint_sha256']}` |")
    lines.append(f"| Checkpoint epoch | {provenance['checkpoint_epoch']} |")
    lines.append(f"| Architecture | {provenance['architecture']} |")
    lines.append(f"| Parameters | {provenance['parameters']:,} |")
    lines.append(f"| Split | `{provenance['split_path']}` |")
    lines.append(f"| Split SHA-256 | `{provenance['split_sha256']}` |")
    lines.append(f"| Samples | {provenance['n_samples']:,} |")
    lines.append(f"| Device | {provenance['device']} |")
    lines.append(f"| Batch size | {provenance['batch_size']} |")
    lines.append(f"| Numeric precision | {provenance['precision']} |")
    lines.append(f"| Generated (UTC) | {provenance['generated_utc']} |")
    lines.append("")
    lines.append("## Headline metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Overall accuracy | {metrics['overall_accuracy']:.6f} |")
    lines.append(f"| Macro F1 | {metrics['macro_f1']:.6f} |")
    lines.append(f"| Minimum class accuracy | {metrics['min_class_accuracy']:.6f} |")
    lines.append(f"| Samples evaluated | {provenance['n_samples']:,} |")
    lines.append("")
    lines.append("## Per-class results")
    lines.append("")
    lines.append("| ID | Class | n | Correct | Accuracy | Precision | Recall | F1 |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
    for class_id in sorted(per_class_metrics):
        m = per_class_metrics[class_id]
        support = int(supports[class_id])
        correct = int(np.diag(confusion)[class_id])
        accuracy = (correct / support) if support else float("nan")
        name = class_names[class_id] if class_id < len(class_names) else str(class_id)
        lines.append(
            f"| {class_id} | {name} | {support} | {correct} | {accuracy:.4f} | "
            f"{m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- The checkpoint was selected on the validation partition by validation macro F1 "
        "with the minimum-class tie-break. This report does not re-select it."
    )
    lines.append(
        "- The inference path runs in float32. The training loop runs under mixed precision, "
        "so stored validation metrics and this report are not expected to agree to the last digit."
    )
    lines.append(
        "- Per-class results are reported in full, for every class, including classes with "
        "very small support."
    )
    lines.append(
        "- This file records measurement only. Whether any difference from another model is "
        "meaningful is decided in the knowledge base, against a stated noise floor."
    )
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen checkpoint on a held-out test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a .pth checkpoint.")
    parser.add_argument("--split", required=True, help="Path to a test split CSV.")
    parser.add_argument("--output", required=True, help="Directory for the report artefacts.")
    parser.add_argument(
        "--label",
        default=None,
        help="Human-readable model name used in the report title.",
    )
    parser.add_argument(
        "--partition-label",
        default="standard test partition",
        help="Name of the evaluated partition, used in figure titles.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the batch size recorded in the checkpoint config.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override the dataloader worker count recorded in the checkpoint config.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device string, e.g. 'cuda' or 'cpu'. Auto-detected if omitted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    split_path = Path(args.split).resolve()
    output_dir = Path(args.output).resolve()

    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")
    if not split_path.exists():
        raise SystemExit(f"Split not found: {split_path}")

    reject_validation_split(split_path)

    # 1. Load the checkpoint and its embedded config.
    payload = load_checkpoint(checkpoint_path, map_location="cpu")
    cfg = payload.get("config")
    if not isinstance(cfg, dict) or not cfg:
        raise SystemExit(
            f"Checkpoint '{checkpoint_path.name}' does not embed a config. "
            "Re-evaluate with a checkpoint written by this project's trainer."
        )

    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    training_cfg = cfg.get("training", {})
    num_classes = int(model_cfg.get("num_classes") or data_cfg.get("num_classes", 17))

    # 2. Rebuild the architecture from the checkpoint config and load the weights.
    model = build_model(cfg)
    model.load_state_dict(payload["model"])
    architecture = str(model_cfg.get("architecture", "sense")).lower()
    parameters = sum(p.numel() for p in model.parameters())

    device = resolve_device(args.device)
    model.to(device)

    # 3. Build the evaluation loader with the training-time preprocessing.
    image_size = int(data_cfg.get("image_size", 224))
    enhancement_method = data_cfg.get("enhancement_method", None)
    batch_size = args.batch_size or int(training_cfg.get("batch_size", 16))
    num_workers = (
        args.num_workers if args.num_workers is not None else int(data_cfg.get("num_workers", 4))
    )

    loader = get_dataloader(
        df_or_path=str(split_path),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        transform=eval_transforms(image_size),
        enhancement_method=enhancement_method,
    )

    n_samples = len(loader.dataset)
    print(
        f"Evaluating {checkpoint_path.name}\n"
        f"  architecture:  {architecture}\n"
        f"  parameters:    {parameters:,}\n"
        f"  split:         {split_path.name} ({n_samples:,} samples)\n"
        f"  device:        {device}\n"
    )

    # 4. Inference and metrics.
    y_true, y_pred = run_inference(model, loader, device)
    if len(y_true) != n_samples:
        raise SystemExit(f"Inference produced {len(y_true)} predictions for {n_samples} samples.")

    metrics = evaluate_predictions(y_true, y_pred, num_classes=num_classes)
    confusion = np.asarray(metrics["confusion_matrix"])
    if int(confusion.sum()) != n_samples:
        raise SystemExit(f"Confusion matrix sums to {int(confusion.sum())}, expected {n_samples}.")
    recomputed_accuracy = float(np.trace(confusion) / confusion.sum())
    if abs(recomputed_accuracy - float(metrics["overall_accuracy"])) > 1e-9:
        raise SystemExit("Reported accuracy does not match the confusion matrix diagonal.")

    per_class_metrics = metrics["per_class_metrics"]
    class_names = list(CADDY_CLASS_NAMES)

    # 5. Provenance block.
    provenance = {
        "checkpoint_path": (
            str(checkpoint_path.relative_to(ROOT))
            if checkpoint_path.is_relative_to(ROOT)
            else str(checkpoint_path)
        ),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": payload.get("epoch"),
        "architecture": architecture,
        "parameters": parameters,
        "split_path": (
            str(split_path.relative_to(ROOT))
            if split_path.is_relative_to(ROOT)
            else str(split_path)
        ),
        "split_sha256": sha256_file(split_path),
        "n_samples": n_samples,
        "device": str(device),
        "batch_size": batch_size,
        "precision": "float32",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "torch_version": torch.__version__,
    }

    label = args.label or (
        "SENSE (selected configuration)" if architecture == "sense" else "Single-image baseline"
    )

    # 6. Write artefacts.
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = {
        "label": label,
        "provenance": provenance,
        "class_names": class_names,
        "checkpoint_stored_metrics": _jsonable(payload.get("metrics", {})),
        "metrics": _jsonable(metrics),
    }
    (output_dir / "metrics.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")

    np.save(output_dir / "confusion_matrix.npy", confusion)
    plot_confusion_matrix(
        confusion,
        class_names=class_names,
        save_path=output_dir / "confusion_matrix.png",
        title=f"Confusion matrix - {label} ({args.partition_label})",
        normalize=False,
    )
    plot_confusion_matrix(
        confusion,
        class_names=class_names,
        save_path=output_dir / "confusion_matrix_norm.png",
        title=f"Row-normalised confusion matrix - {label} ({args.partition_label})",
        normalize=True,
    )
    write_per_class_csv(output_dir / "per_class.csv", per_class_metrics, confusion, class_names)
    write_report(
        output_dir / "report.md", label, provenance, metrics, per_class_metrics, class_names
    )

    # 7. Console summary.
    print(
        f"\nResults ({n_samples:,} samples)\n"
        f"  overall accuracy:   {metrics['overall_accuracy']:.6f}\n"
        f"  macro F1:           {metrics['macro_f1']:.6f}\n"
        f"  min class accuracy: {metrics['min_class_accuracy']:.6f}\n"
        f"  report directory:   {output_dir}\n"
    )


if __name__ == "__main__":
    main()
