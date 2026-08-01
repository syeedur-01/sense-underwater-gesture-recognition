"""
scripts/train_baseline.py
Phase 4 — P4.2: Minimal training loop for the ResNet18 single-image baseline.

Usage (from project root, with venv active):
    python scripts/train_baseline.py

Config: configs/baseline_transfer_learning.yaml (merged over configs/base.yaml)
Output:
    checkpoints/baseline_best.pth   — best val-accuracy checkpoint
    runs/baseline/metrics.csv        — per-epoch metrics log
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# ── project root on sys.path so `src` imports work ─────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import Paths, load_yaml_config
from src.data.caddy_dataset import get_dataloader
from src.data.transforms import eval_transforms, train_transforms
from src.models.visual_encoder import SingleImageClassifier

# ──────────────────────────────────────────────────────────────────────────────


def merge_configs(base_path: Path, override_path: Path) -> dict:
    """Load base YAML and shallow-merge override on top of it."""
    base = load_yaml_config(str(base_path))
    override = load_yaml_config(str(override_path))

    def _deep_merge(b: dict, o: dict) -> dict:
        result = dict(b)
        for key, val in o.items():
            if key == "_base_":
                continue  # skip the inheritance directive
            if isinstance(val, dict) and isinstance(result.get(key), dict):
                result[key] = _deep_merge(result[key], val)
            else:
                result[key] = val
        return result

    return _deep_merge(base, override)


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Top-1 accuracy (0.0–1.0)."""
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool = True,
) -> tuple[float, float]:
    """Run one epoch; returns (avg_loss, avg_accuracy)."""
    model.train(train)
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            imgs = batch["left"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            logits = model(imgs)
            loss = criterion(logits, labels)

            if train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_acc += accuracy(logits, labels)
            n_batches += 1

    return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Phase 4 ResNet18 baseline.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "baseline_transfer_learning.yaml"),
        help="Path to baseline YAML config.",
    )
    args = parser.parse_args()

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = merge_configs(
        ROOT / "configs" / "base.yaml",
        Path(args.config),
    )

    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_baseline] Device: {device}")

    image_size: int = data_cfg.get("image_size", 224)
    batch_size: int = train_cfg.get("batch_size", 32)
    num_epochs: int = train_cfg.get("epochs", 20)
    base_lr: float = train_cfg.get("base_lr", 1e-3)

    # Split paths — allow override from config
    train_csv = Path(data_cfg.get("split_train", Paths.SPLITS_DIR / "standard_train.csv"))
    val_csv = Path(data_cfg.get("split_val", Paths.SPLITS_DIR / "standard_val.csv"))

    # ── Data ──────────────────────────────────────────────────────────────────
    t_train = train_transforms(image_size)
    t_eval = eval_transforms(image_size)

    # num_workers=0 on Windows to avoid multiprocessing issues with CUDA
    nw = 0 if sys.platform == "win32" else 4

    train_loader = get_dataloader(
        train_csv,
        batch_size=batch_size,
        shuffle=True,
        num_workers=nw,
        transform=t_train,
        pin_memory=device.type == "cuda",
    )
    val_loader = get_dataloader(
        val_csv,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nw,
        transform=t_eval,
        pin_memory=device.type == "cuda",
    )

    print(f"[train_baseline] Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    backbone: str = model_cfg.get("backbone", "resnet18")
    num_classes: int = model_cfg.get("num_classes", 17)
    pretrained: bool = model_cfg.get("pretrained", True)

    model = SingleImageClassifier(
        backbone=backbone,
        num_classes=num_classes,
        pretrained=pretrained,
    ).to(device)

    # Phase schedule from config
    phases_cfg = train_cfg.get("phases", {})
    phase1_end = phases_cfg.get("phase1", {}).get("epochs", [1, 5])[-1]
    phase2_lr_scale = phases_cfg.get("phase2", {}).get("lr_scale", 0.1)

    # ── Loss (plain unweighted CE for baseline comparability) ─────────────────
    criterion = nn.CrossEntropyLoss()

    # ── Optimizer + Scheduler (phase 1: backbone frozen) ──────────────────────
    model.freeze_backbone()
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=base_lr,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # ── Checkpoints + Metrics dirs ────────────────────────────────────────────
    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / "baseline_best.pth"

    runs_dir = ROOT / "runs" / "baseline"
    runs_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = runs_dir / "metrics.csv"

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc = 0.0
    csv_header = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"]

    with open(metrics_path, "w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(csv_header)

        for epoch in range(1, num_epochs + 1):
            # ── Phase 2: unfreeze backbone at epoch phase1_end + 1 ─────────
            if epoch == phase1_end + 1:
                print(f"[train_baseline] Epoch {epoch}: unfreezing backbone (phase 2)")
                model.unfreeze_backbone()
                # Rebuild optimizer with scaled LR for backbone params
                optimizer = AdamW(model.parameters(), lr=base_lr * phase2_lr_scale)
                scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - phase1_end)

            # ── Train ─────────────────────────────────────────────────────────
            train_loss, train_acc = run_epoch(
                model, train_loader, criterion, optimizer, device, train=True
            )
            scheduler.step()

            # ── Validate ──────────────────────────────────────────────────────
            val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device, train=False)

            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:03d}/{num_epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
                f"lr={current_lr:.2e}"
            )
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, current_lr])
            f_csv.flush()

            # ── Save best ─────────────────────────────────────────────────────
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "val_acc": val_acc,
                        "val_loss": val_loss,
                        "backbone": backbone,
                        "num_classes": num_classes,
                    },
                    best_ckpt,
                )
                print(f"  [+] New best val_acc={val_acc:.4f} -- checkpoint saved.")

    print(f"\n[train_baseline] Training complete. Best val_acc={best_val_acc:.4f}")
    print(f"  Checkpoint : {best_ckpt}")
    print(f"  Metrics CSV: {metrics_path}")


if __name__ == "__main__":
    main()
