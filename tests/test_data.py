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
    train_tf = train_transforms(image_size=224)
    augmented = train_tf(image=img_left, image_right=img_right)

    left_t = augmented["image"]
    right_t = augmented["image_right"]

    assert left_t.shape == (3, 224, 224)
    assert right_t.shape == (3, 224, 224)
    assert torch.equal(
        left_t, right_t
    ), "Stereo augmentation parameters differed between left and right images!"

    # 2. Eval transforms: verify determinism across repeated calls
    eval_tf = eval_transforms(image_size=224)
    out1 = eval_tf(image=img_left, image_right=img_right)
    out2 = eval_tf(image=img_left, image_right=img_right)

    assert torch.equal(out1["image"], out2["image"]), "Eval transforms are not deterministic!"
    assert torch.equal(
        out1["image_right"], out2["image_right"]
    ), "Eval transforms are not deterministic!"


def test_dataloader_batch_iteration(sample_manifest_df: pd.DataFrame):
    """P2.6 Verification: verify get_dataloader iterates batches with expected tensor shapes."""
    caddy_root = Path(Paths.CADDY_ROOT)
    if not caddy_root.exists():
        pytest.skip(f"CADDY_ROOT not found at {caddy_root}")

    from src.data.caddy_dataset import get_dataloader
    from src.data.transforms import eval_transforms

    loader = get_dataloader(
        df_or_path=sample_manifest_df,
        batch_size=4,
        shuffle=False,
        num_workers=0,  # Single-process for unit test
        transform=eval_transforms(image_size=224),
        caddy_root=caddy_root,
    )

    batch = next(iter(loader))

    assert "left" in batch and "right" in batch
    assert "label" in batch and "scenario_id" in batch and "sample_id" in batch

    b_size = min(4, len(sample_manifest_df))
    assert batch["left"].shape == (b_size, 3, 224, 224)
    assert batch["right"].shape == (b_size, 3, 224, 224)
    assert batch["label"].shape == (b_size,)
    assert batch["scenario_id"].shape == (b_size,)
    assert len(batch["sample_id"]) == b_size


def test_class_weight_computation():
    """P2.8 Verification: verify compute_class_weights on synthetic imbalanced and balanced data."""
    from src.data.splits import compute_class_weights

    # 1. Imbalanced data: minority class (1) gets higher weight than majority class (0)
    df_imbalanced = pd.DataFrame({"gesture_id": [0] * 100 + [1] * 10})
    w_imb = compute_class_weights(df_imbalanced, num_classes=2)

    assert isinstance(w_imb, torch.FloatTensor)
    assert len(w_imb) == 2
    assert (w_imb > 0).all() and torch.isfinite(w_imb).all()
    assert w_imb[1] > w_imb[0]  # Minority class receives higher weight

    # 2. Balanced data: weights are uniform (≈1.0)
    df_balanced = pd.DataFrame({"gesture_id": [0] * 50 + [1] * 50})
    w_bal = compute_class_weights(df_balanced, num_classes=2)

    assert torch.allclose(w_bal, torch.tensor([1.0, 1.0], dtype=torch.float32), atol=1e-4)


def test_manifest_class_count_matches_config():
    """Fix 1 verification: manifest unique gesture_ids == 17 (matches num_classes in configs)."""
    manifest_path = Path("data/processed/manifest.csv")
    if not manifest_path.exists():
        pytest.skip("manifest.csv not present — run scripts/build_manifest.py first")
    df = pd.read_csv(manifest_path)
    unique_classes = df["gesture_id"].nunique()
    assert unique_classes == 17, (
        f"Expected 17 unique gesture_ids in manifest (16 gestures + true_neg), "
        f"found {unique_classes}"
    )
