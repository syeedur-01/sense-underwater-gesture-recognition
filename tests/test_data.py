"""
tests/test_data.py
Unit tests for Phase 2 data pipeline (CaddyStereoDataset and data splits).
"""

from pathlib import Path

import pandas as pd
import pytest
import torch

from src.config import Paths
from src.data.caddy_dataset import CaddyStereoDataset


@pytest.fixture
def sample_manifest_df() -> pd.DataFrame:
    """Fixture providing a mini manifest DataFrame for testing."""
    manifest_path = Path("data/processed/manifest.csv")
    if manifest_path.exists():
        df = pd.read_csv(manifest_path)
        return df.head(10).copy()

    # Fallback dummy data if manifest.csv is missing
    return pd.DataFrame(
        {
            "sample_id": ["pos_00000", "neg_00000"],
            "left_path": [
                "biograd-A/true_positives/raw/biograd-A_00000_left.jpg",
                "biograd-A/true_negatives/raw/biograd-A_00000_left.jpg",
            ],
            "right_path": [
                "biograd-A/true_positives/raw/biograd-A_00000_right.jpg",
                "biograd-A/true_negatives/raw/biograd-A_00000_right.jpg",
            ],
            "gesture_label": ["num_delimiter", "true_neg"],
            "gesture_id": [10, 16],
            "scenario_id": [0, 0],
            "scenario": ["biograd-A", "biograd-A"],
            "is_negative": [0, 1],
        }
    )


def test_dataset_getitem_shapes(sample_manifest_df: pd.DataFrame):
    """P2.4 Verification: verify dataset output dictionary keys, tensor shapes, and label ranges."""
    caddy_root = Path(Paths.CADDY_ROOT)
    if not caddy_root.exists():
        pytest.skip(f"CADDY_ROOT not found at {caddy_root}")

    dataset = CaddyStereoDataset(df_or_path=sample_manifest_df, caddy_root=caddy_root)
    assert len(dataset) == len(sample_manifest_df)

    item = dataset[0]

    # Verify return dictionary keys
    expected_keys = {"left", "right", "label", "scenario_id", "sample_id"}
    assert set(item.keys()) == expected_keys

    # Verify tensor types and shapes
    assert isinstance(item["left"], torch.Tensor)
    assert isinstance(item["right"], torch.Tensor)
    assert item["left"].ndim == 3
    assert item["right"].ndim == 3
    assert item["left"].shape[0] == 3  # RGB channels
    assert item["left"].shape == item["right"].shape

    # Verify label range (0..16)
    assert isinstance(item["label"], int)
    assert 0 <= item["label"] <= 16

    # Verify scenario ID range (0..7)
    assert isinstance(item["scenario_id"], int)
    assert 0 <= item["scenario_id"] <= 7

    # Verify sample ID string
    assert isinstance(item["sample_id"], str)
    assert item["sample_id"].startswith(("pos_", "neg_"))


def test_stereo_augmentation_consistency():
    """P2.5 Verification: verify stereo transform parameter identity and eval determinism."""
    import numpy as np

    from src.data.transforms import eval_transforms, train_transforms

    np.random.seed(42)
    # Generate identical left and right test image arrays
    img_left = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    img_right = img_left.copy()

    # 1. Train transforms: verify exact identity on identical stereo inputs
    train_tf = train_transforms(image_size=320)
    augmented = train_tf(image=img_left, image_right=img_right)

    left_t = augmented["image"]
    right_t = augmented["image_right"]

    assert left_t.shape == (3, 320, 320)
    assert right_t.shape == (3, 320, 320)
    assert torch.equal(
        left_t, right_t
    ), "Stereo augmentation parameters differed between left and right images!"

    # 2. Eval transforms: verify determinism across repeated calls
    eval_tf = eval_transforms(image_size=320)
    out1 = eval_tf(image=img_left, image_right=img_right)
    out2 = eval_tf(image=img_left, image_right=img_right)

    assert torch.equal(out1["image"], out2["image"]), "Eval transforms are not deterministic!"
    assert torch.equal(
        out1["image_right"], out2["image_right"]
    ), "Eval transforms are not deterministic!"
