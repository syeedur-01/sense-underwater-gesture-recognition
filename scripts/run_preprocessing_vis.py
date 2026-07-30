"""
scripts/run_preprocessing_vis.py
P2.7 — Generates 5 sample raw vs. augmented stereo pair visualizations and RGB histograms.

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
    print("SENSE — P2.7 Preprocessing & Augmentation Visualization (5 Samples)")
    print("=" * 60)

    manifest_path = PROCESSED_DIR / "manifest.csv"
    if not manifest_path.exists():
        from src.data.splits import build_manifest

        df = build_manifest(output_path=manifest_path)
    else:
        df = pd.read_csv(manifest_path)

    # Sample 5 distinct positive gesture samples across different scenarios
    pos_df = df[df["is_negative"] == 0]
    samples = pos_df.sample(5, random_state=42).reset_index(drop=True)

    train_tf = train_transforms(image_size=224)
    eval_tf = eval_transforms(image_size=224)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # ----------------------------------------------------------------
    # Chart 1: 5-Sample Grid (Raw Left | Aug Left | Aug Right)
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(5, 3, figsize=(12, 16))
    fig.suptitle(
        "P2.7 — Stereo Preprocessing QA: 5 Random Samples\n"
        "Raw (640x480) vs. Stereo Augmented Pairs (224x224)",
        fontsize=13,
        y=0.995,
    )

    for row_idx, (_, row) in enumerate(samples.iterrows()):
        left_path = CADDY_ROOT / str(row["left_path"]).lstrip("/")
        right_path = CADDY_ROOT / str(row["right_path"]).lstrip("/")

        img_left = np.array(Image.open(left_path).convert("RGB"))
        img_right = np.array(Image.open(right_path).convert("RGB"))

        aug = train_tf(image=img_left, image_right=img_right)
        aug_left = aug["image"].numpy().transpose(1, 2, 0)
        aug_right = aug["image_right"].numpy().transpose(1, 2, 0)

        aug_left_disp = np.clip(aug_left * std + mean, 0, 1)
        aug_right_disp = np.clip(aug_right * std + mean, 0, 1)

        info_text = (
            f"Sample {row_idx+1}: {row['sample_id']} | {row['scenario']} | "
            f"Label: {row['gesture_label']}"
        )

        # Column 1: Raw Left Image
        axes[row_idx, 0].imshow(img_left)
        axes[row_idx, 0].set_title(f"RAW LEFT (640x480)\n{info_text}", fontsize=8)
        axes[row_idx, 0].axis("off")

        # Column 2: Augmented Left Image
        axes[row_idx, 1].imshow(aug_left_disp)
        axes[row_idx, 1].set_title(f"AUGMENTED LEFT (224x224)\n{row['gesture_label']}", fontsize=8)
        axes[row_idx, 1].axis("off")

        # Column 3: Augmented Right Image
        axes[row_idx, 2].imshow(aug_right_disp)
        axes[row_idx, 2].set_title(f"AUGMENTED RIGHT (224x224)\n{row['gesture_label']}", fontsize=8)
        axes[row_idx, 2].axis("off")

    plt.tight_layout()
    out1 = PROCESSED_DIR / "preprocessing_raw_vs_augmented.png"
    plt.savefig(out1, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved 5-sample grid: {out1}")

    # ----------------------------------------------------------------
    # Chart 2: Pixel Intensity Histograms (First Sample)
    # ----------------------------------------------------------------
    sample0 = samples.iloc[0]
    left0 = CADDY_ROOT / str(sample0["left_path"]).lstrip("/")
    right0 = CADDY_ROOT / str(sample0["right_path"]).lstrip("/")
    img_left0 = np.array(Image.open(left0).convert("RGB"))
    img_right0 = np.array(Image.open(right0).convert("RGB"))

    aug0 = train_tf(image=img_left0, image_right=img_right0)
    aug_left0 = aug0["image"].numpy().transpose(1, 2, 0)
    eval_out0 = eval_tf(image=img_left0, image_right=img_right0)
    eval_left0 = eval_out0["image"].numpy().transpose(1, 2, 0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = ["red", "green", "blue"]

    for i, c in enumerate(colors):
        axes[0].hist(img_left0[:, :, i].ravel(), bins=50, color=c, alpha=0.5, label=f"{c.upper()}")
    axes[0].set_title("Raw Image RGB Histogram (0..255)")
    axes[0].set_xlabel("Pixel Value")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    for i, c in enumerate(colors):
        axes[1].hist(eval_left0[:, :, i].ravel(), bins=50, color=c, alpha=0.5, label=f"{c.upper()}")
    axes[1].set_title("Eval Preprocessed (Normalized ImageNet mean/std)")
    axes[1].set_xlabel("Standardized Value")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()

    for i, c in enumerate(colors):
        axes[2].hist(aug_left0[:, :, i].ravel(), bins=50, color=c, alpha=0.5, label=f"{c.upper()}")
    axes[2].set_title("Train Augmented (ColorJitter + Blur + Noise + Norm)")
    axes[2].set_xlabel("Standardized Value")
    axes[2].set_ylabel("Frequency")
    axes[2].legend()

    plt.tight_layout()
    out2 = PROCESSED_DIR / "preprocessing_rgb_histograms.png"
    plt.savefig(out2, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved histograms: {out2}")

    print("\n" + "=" * 60)
    print("P2.7 PREPROCESSING VISUALIZATION COMPLETE (5 SAMPLES)")
    print("=" * 60)


if __name__ == "__main__":
    main()
