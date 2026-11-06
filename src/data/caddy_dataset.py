"""
src/data/caddy_dataset.py
PyTorch Dataset implementation for CADDY stereo gesture recognition.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import Paths

# Valid enhancement method identifiers (Phase 3 / P3.7).
VALID_ENHANCEMENT_METHODS = {"none", "funie_gan", "waternet"}


class CaddyStereoDataset(Dataset):
    """
    PyTorch Dataset for CADDY stereo diver gesture recognition.

    Yields a dictionary for each stereo pair:
        {
            "left": torch.Tensor (C, H, W),
            "right": torch.Tensor (C, H, W),
            "label": int (0..16),
            "scenario_id": int (0..7),
            "sample_id": str,
        }

    Enhancement (P3.7):
        Pass ``enhancement_method`` (one of "none" / "funie_gan" / "waternet")
        to automatically resolve the correct cached-image subdirectory under
        ``Paths.ENHANCED_CACHE_DIR``.  Alternatively, supply ``enhanced_dir``
        directly.  If both are given, ``enhanced_dir`` takes precedence.
        "none" means: use the raw CADDY images (no subdirectory lookup).
    """

    def __init__(
        self,
        df_or_path: pd.DataFrame | str | Path,
        caddy_root: str | Path | None = None,
        transform: Callable | None = None,
        enhanced_dir: str | Path | None = None,
        enhancement_method: str | None = None,
    ) -> None:
        """
        Args:
            df_or_path: DataFrame or Path/str to a split CSV manifest.
            caddy_root: Root directory of extracted CADDY raw dataset.
            transform: Albumentations/callable transform pipeline.
            enhanced_dir: Directory containing cached enhanced images (Phase 3).
                          Takes precedence over ``enhancement_method`` if supplied.
            enhancement_method: One of "none", "funie_gan", "waternet".
                                 Resolved to ``{Paths.ENHANCED_CACHE_DIR}/{method}``.
                                 Ignored when ``enhanced_dir`` is explicitly provided.
                                 "none" (or None) → load raw images from CADDY_ROOT.
        """
        if isinstance(df_or_path, (str, Path)):
            self.df = pd.read_csv(df_or_path)
        else:
            self.df = df_or_path.reset_index(drop=True)

        self.caddy_root = Path(caddy_root or Paths.CADDY_ROOT)
        self.transform = transform

        # P3.7 — resolve enhanced_dir from config flag when not given directly.
        if enhanced_dir is not None:
            self.enhanced_dir: Path | None = Path(enhanced_dir)
        elif enhancement_method is not None and enhancement_method != "none":
            if enhancement_method not in VALID_ENHANCEMENT_METHODS:
                raise ValueError(
                    f"Unknown enhancement_method {enhancement_method!r}. "
                    f"Must be one of: {sorted(VALID_ENHANCEMENT_METHODS)}"
                )
            self.enhanced_dir = Path(Paths.ENHANCED_CACHE_DIR) / enhancement_method
        else:
            self.enhanced_dir = None

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, rel_path: str) -> np.ndarray:
        """Load single image as RGB numpy array (H, W, 3) from raw CADDY root."""
        rel_str = rel_path.lstrip("/")
        raw_path = self.caddy_root / rel_str
        if not raw_path.exists():
            raise FileNotFoundError(f"Image not found at path: {raw_path}")
        img = cv2.imread(str(raw_path))
        if img is None:
            raise ValueError(f"Failed to read image at path: {raw_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _load_enhanced(self, sample_id: str, side: str) -> np.ndarray | None:
        """
        Try to load a cached enhanced image keyed by sample_id and side.

        P3.5 cache naming: ``{enhanced_dir}/{sample_id}_{side}.png``
        Returns None if the cache directory is not configured or file is missing.
        """
        if self.enhanced_dir is None:
            return None
        cached_path = self.enhanced_dir / f"{sample_id}_{side}.png"
        if not cached_path.exists():
            return None
        img = cv2.imread(str(cached_path))
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        sample_id = str(row["sample_id"])

        left_rel = str(row["left_path"])
        right_rel = str(row["right_path"])

        # Prefer cached enhanced image; fall back to raw.
        img_left = self._load_enhanced(sample_id, "left") or self._load_image(left_rel)
        img_right = self._load_enhanced(sample_id, "right") or self._load_image(right_rel)

        if self.transform is not None:
            # Albumentations pipeline expecting image and image_right
            transformed = self.transform(image=img_left, image_right=img_right)
            tensor_left = transformed["image"]
            tensor_right = transformed["image_right"]
        else:
            # Default to float tensor (C, H, W) normalized to [0, 1]
            tensor_left = torch.from_numpy(img_left.transpose(2, 0, 1)).float() / 255.0
            tensor_right = torch.from_numpy(img_right.transpose(2, 0, 1)).float() / 255.0

        return {
            "left": tensor_left,
            "right": tensor_right,
            "label": int(row["gesture_id"]),
            "scenario_id": int(row["scenario_id"]),
            "sample_id": str(row["sample_id"]),
        }


def get_dataloader(
    df_or_path: pd.DataFrame | str | Path,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    transform: Callable | None = None,
    caddy_root: str | Path | None = None,
    enhanced_dir: str | Path | None = None,
    enhancement_method: str | None = None,
    pin_memory: bool = True,
    sampler: torch.utils.data.Sampler | None = None,
) -> torch.utils.data.DataLoader:
    """
    DataLoader factory function for CaddyStereoDataset.

    Args:
        df_or_path: DataFrame or Path/str to a split CSV manifest.
        batch_size: Batch size for DataLoader.
        shuffle: Whether to shuffle samples (ignored if sampler is provided).
        num_workers: Number of worker processes.
        transform: Albumentations or callable transform pipeline.
        caddy_root: Root directory of extracted CADDY raw dataset.
        enhanced_dir: Directory containing cached enhanced images (Phase 3).
                      Takes precedence over ``enhancement_method`` if supplied.
        enhancement_method: One of "none", "funie_gan", "waternet" (P3.7).
                            Resolved to ``{Paths.ENHANCED_CACHE_DIR}/{method}``.
        pin_memory: If True, copies Tensors into CUDA pinned memory.
        sampler: Optional Sampler (e.g. WeightedRandomSampler for P2.9).

    Returns:
        Configured PyTorch DataLoader instance.
    """
    dataset = CaddyStereoDataset(
        df_or_path=df_or_path,
        caddy_root=caddy_root,
        transform=transform,
        enhanced_dir=enhanced_dir,
        enhancement_method=enhancement_method,
    )

    should_shuffle = shuffle if sampler is None else False

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=should_shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
