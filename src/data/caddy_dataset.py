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
    """

    def __init__(
        self,
        df_or_path: pd.DataFrame | str | Path,
        caddy_root: str | Path | None = None,
        transform: Callable | None = None,
        enhanced_dir: str | Path | None = None,
    ) -> None:
        """
        Args:
            df_or_path: DataFrame or Path/str to a split CSV manifest.
            caddy_root: Root directory of extracted CADDY raw dataset.
            transform: Albumentations/callable transform pipeline.
            enhanced_dir: Directory containing cached enhanced images (Phase 3).
        """
        if isinstance(df_or_path, (str, Path)):
            self.df = pd.read_csv(df_or_path)
        else:
            self.df = df_or_path.reset_index(drop=True)

        self.caddy_root = Path(caddy_root or Paths.CADDY_ROOT)
        self.transform = transform
        self.enhanced_dir = Path(enhanced_dir) if enhanced_dir else None

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, rel_path: str) -> np.ndarray:
        """Load single image as RGB numpy array (H, W, 3)."""
        rel_str = rel_path.lstrip("/")

        # Try cached enhanced image first if enhanced_dir is provided
        if self.enhanced_dir is not None:
            enhanced_path = self.enhanced_dir / rel_str
            if enhanced_path.exists():
                img = cv2.imread(str(enhanced_path))
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Fallback to raw CADDY root image
        raw_path = self.caddy_root / rel_str
        if not raw_path.exists():
            raise FileNotFoundError(f"Image not found at path: {raw_path}")

        img = cv2.imread(str(raw_path))
        if img is None:
            raise ValueError(f"Failed to read image at path: {raw_path}")

        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]

        left_rel = str(row["left_path"])
        right_rel = str(row["right_path"])

        img_left = self._load_image(left_rel)
        img_right = self._load_image(right_rel)

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
