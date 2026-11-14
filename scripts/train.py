"""
scripts/train.py
================
P6.9 — Command-line training entry point for SENSE.

Usage (from project root with venv active):
    python scripts/train.py --config configs/sense_full.yaml
    python scripts/train.py --config configs/sense_full.yaml --epochs 1
    python scripts/train.py --config configs/sense_full.yaml --resume runs/.../checkpoints/last.pth
    python scripts/train.py --config configs/sense_full.yaml --experiment-name my_run

The script:
  1. Loads and merges the YAML config (with base.yaml inheritance).
  2. Seeds RNGs via set_seed().
  3. Builds SENSE from config.
  4. Runs Trainer.fit() — training on train split, evaluation on val split only.
  5. Produces a full experiment record (all nine P6.5 artifacts) in
     runs/{experiment_name}/{timestamp}/.

Invariant: This script never reads or writes any test-split CSV.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# -- project root on sys.path so `src` imports work from any CWD ----
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.config import _deep_merge, load_merged_config  # noqa: E402
from src.models.sense import SENSE  # noqa: E402
from src.training.trainer import Trainer  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

__all__ = ["_deep_merge", "load_merged_config", "main"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train SENSE gesture recognition model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config",
        required=True,
        help="Path to experiment YAML config (e.g. configs/sense_full.yaml).",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training.epochs from config (useful for smoke tests).",
    )
    p.add_argument(
        "--resume",
        default=None,
        help="Path to a checkpoint to resume training from.",
    )
    p.add_argument(
        "--experiment-name",
        default=None,
        dest="experiment_name",
        help="Logical name for this run (default: stem of config filename).",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Torch device string, e.g. 'cuda' or 'cpu'. Auto-detected if omitted.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # 1. Load config
    cfg = load_merged_config(args.config)

    # 2. Override epochs if requested (e.g. smoke test: --epochs 1)
    if args.epochs is not None:
        cfg.setdefault("training", {})["epochs"] = args.epochs

    # 3. Seed — set early, before any model/data construction
    seed: int = int(cfg.get("training", {}).get("seed", 42))
    set_seed(seed)

    # 4. Resolve experiment name
    experiment_name = args.experiment_name or Path(args.config).stem

    # 5. Resolve data CSVs — from config, relative to project root
    data_cfg = cfg.get("data", {})
    train_csv = Path(ROOT) / data_cfg["split_train"]
    val_csv = Path(ROOT) / data_cfg["split_val"]

    # Guard: ensure test CSV is NOT passed to Trainer
    # (The key may exist in config for documentation purposes; we simply
    #  don't pass it here — it's only used in Phase 8 evaluation.)

    # 6. Build model
    model = SENSE.from_config(cfg)

    # 7. Device
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 8. Build Trainer and run
    trainer = Trainer(
        model=model,
        cfg=cfg,
        train_csv=str(train_csv),
        val_csv=str(val_csv),
        experiment_name=experiment_name,
        device=device,
        resume_path=args.resume,
    )

    best_metrics = trainer.fit()

    min_cls = best_metrics.get("min_class_accuracy", float("nan"))
    print(
        f"\nTraining complete.\n"
        f"  Best val macro F1:       {best_metrics.get('macro_f1', float('nan')):.4f}\n"
        f"  Best val accuracy:       {best_metrics.get('accuracy', float('nan')):.4f}\n"
        f"  Best val min-class acc:  {min_cls:.4f}\n"
        f"  Experiment record:       {trainer.tracker.run_directory}\n"
    )


if __name__ == "__main__":
    main()
