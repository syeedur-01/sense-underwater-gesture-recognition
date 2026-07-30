"""
scripts/run_preprocessing_vis.py
P2.7 — Generates raw vs. augmented stereo pair visualizations and RGB histograms.

Outputs:
    data/processed/preprocessing_raw_vs_augmented.png
    data/processed/preprocessing_rgb_histograms.png
"""

import io
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import matplotlib

matplotlib.use("Agg")  # Headless mode
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import Paths
from src.data.transforms import eval_transforms, train_transforms

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
CADDY_ROOT = Path(Paths.CADDY_ROOT)


def main():
    print("=" * 60)
    print("SENSE — P2.7 Preprocessing & Augmentation Visualization")
    print("=" * 60)

    manifest_path = PROCESSED_DIR / "manifest.csv"
    if not manifest_path.exists():
        from src.data.splits import build_manifest

        df = build_manifest(output_path=manifest_path)
    else:
        df = pd.read_csv(manifest_path)

    # Pick a representative positive sample (e.g. num_delimiter or photo)
    sample = df[df["is_negative"] == 0].iloc[0]
    left_path = CADDY_ROOT / str(sample["left_path"]).lstrip("/")
    right_path = CADDY_ROOT / str(sample["right_path"]).lstrip("/")

    img_left = np.array(Image.open(left_path).convert("RGB"))
    img_right = np.array(Image.open(right_path).convert("RGB"))

    # Apply train augmentation
    train_tf = train_transforms(image_size=240)
    aug = train_tf(image=img_left, image_right=img_right)
    aug_left = aug["image"].numpy().transpose(1, 2, 0)
    aug_right = aug["image_right"].numpy().transpose(1, 2, 0)

    # Denormalize for display
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    aug_left_disp = np.clip(aug_left * std + mean, 0, 1)
    aug_right_disp = np.clip(aug_right * std + mean, 0, 1)

    # Apply eval preprocessing
    eval_tf = eval_transforms(image_size=240)
    eval_out = eval_tf(image=img_left, image_right=img_right)
    eval_left = eval_out["image"].numpy().transpose(1, 2, 0)

    # ----------------------------------------------------------------
    # Chart 1: Raw vs Augmented Stereo Pair
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(
        f"P2.7 — Stereo Pair Preprocessing: Raw (640x480) vs. Augmented (240x240)\n"
        f"Sample: {sample['sample_id']} | Scenario: {sample['scenario']} | "
        f"Label: {sample['gesture_label']}",
        fontsize=11,
    )

    axes[0, 0].imshow(img_left)
    axes[0, 0].set_title("Raw Left Image (640x480)", fontsize=10)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(img_right)
    axes[0, 1].set_title("Raw Right Image (640x480)", fontsize=10)
    axes[0, 1].axis("off")

    axes[1, 0].imshow(aug_left_disp)
    axes[1, 0].set_title("Augmented Left Image (240x240)", fontsize=10)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(aug_right_disp)
    axes[1, 1].set_title("Augmented Right Image (240x240)", fontsize=10)
    axes[1, 1].axis("off")

    plt.tight_layout()
    out1 = PROCESSED_DIR / "preprocessing_raw_vs_augmented.png"
    plt.savefig(out1, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out1}")

    # ----------------------------------------------------------------
    # Chart 2: Pixel Intensity Histograms (Raw vs Normalized Tensor)
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = ["red", "green", "blue"]

    # Raw pixel values (0..255)
    for i, c in enumerate(colors):
        axes[0].hist(img_left[:, :, i].ravel(), bins=50, color=c, alpha=0.5, label=f"{c.upper()}")
    axes[0].set_title("Raw Image RGB Histogram (0..255)")
    axes[0].set_xlabel("Pixel Value")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    # Eval preprocessed tensor (Normalized)
    for i, c in enumerate(colors):
        axes[1].hist(eval_left[:, :, i].ravel(), bins=50, color=c, alpha=0.5, label=f"{c.upper()}")
    axes[1].set_title("Eval Preprocessed (Normalized ImageNet mean/std)")
    axes[1].set_xlabel("Standardized Value")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()

    # Train augmented tensor (Normalized)
    for i, c in enumerate(colors):
        axes[2].hist(aug_left[:, :, i].ravel(), bins=50, color=c, alpha=0.5, label=f"{c.upper()}")
    axes[2].set_title("Train Augmented (ColorJitter + Blur + Noise + Norm)")
    axes[2].set_xlabel("Standardized Value")
    axes[2].set_ylabel("Frequency")
    axes[2].legend()

    plt.tight_layout()
    out2 = PROCESSED_DIR / "preprocessing_rgb_histograms.png"
    plt.savefig(out2, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out2}")

    print("\n" + "=" * 60)
    print("P2.7 PREPROCESSING VISUALIZATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
