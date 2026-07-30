"""
src/data/transforms.py
Stereo-consistent data augmentation pipelines (Phase 2.5).

Deliberately Excluded Augmentations Rationale:
1. Horizontal flip: Risks corrupting direction-dependent CADDIAN gesture labels
   (e.g., directional sign signals) and breaks physical stereo camera baseline geometry
   (left vs right displacement).
2. CLAHE or contrast/enhancement operations: Excluded from on-the-fly augmentation to
   avoid contaminating the 'none' enhancement baseline arm of Experiment 3/H2. All
   underwater enhancement is strictly isolated to Phase 3 preprocessing (FUnIE-GAN / WaterNet).
3. Synthetic underwater color-cast jitter & cutout/random erasing: Deferred to prevent
   confounding enhancement ablation experiments and to prevent erasing small gesturing hand regions.
4. Class-conditional augmentation intensity: Rejected in favor of a uniform policy across all
   classes to avoid interacting class-frequency mechanisms. Class imbalance is handled in P2.9.
"""

from __future__ import annotations

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2


def train_transforms(
    image_size: tuple[int, int] | int = 240,
) -> A.Compose:
    """
    Returns stereo-consistent training augmentation pipeline.

    Uses additional_targets={"image_right": "image"} so that identical spatial and
    photometric transform parameters are sampled once and applied to both left and right images.
    """
    if isinstance(image_size, int):
        h, w = image_size, image_size
    else:
        h, w = image_size

    return A.Compose(
        [
            A.RandomResizedCrop(size=(h, w), scale=(0.85, 1.0), ratio=(0.9, 1.1)),
            A.Rotate(limit=12, border_mode=cv2.BORDER_CONSTANT, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.15),
            A.GaussNoise(std_range=(0.02, 0.08), p=0.15),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        additional_targets={"image_right": "image"},
    )


def eval_transforms(
    image_size: tuple[int, int] | int = 240,
) -> A.Compose:
    """
    Returns deterministic evaluation preprocessing pipeline (Resize -> Normalize -> ToTensorV2).
    """
    if isinstance(image_size, int):
        h, w = image_size, image_size
    else:
        h, w = image_size

    return A.Compose(
        [
            A.Resize(height=h, width=w),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        additional_targets={"image_right": "image"},
    )
