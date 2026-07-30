"""
src/data/splits.py
Data manifest builder and dataset splitting utility (Phase 2).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path if missing
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pandas as pd  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from src.config import SEED, Paths  # noqa: E402

SCENARIOS = [
    "biograd-A",
    "biograd-B",
    "biograd-C",
    "brodarski-A",
    "brodarski-B",
    "brodarski-C",
    "brodarski-D",
    "genova-A",
]

SCENARIO_TO_ID = {s: i for i, s in enumerate(SCENARIOS)}


def build_manifest(
    positives_csv: Path | str | None = None,
    negatives_csv: Path | str | None = None,
    output_path: Path | str | None = None,
) -> pd.DataFrame:
    """
    Build unified canonical data manifest from CADDY raw CSVs (synthetic == 0 only).

    Returns DataFrame with columns:
        sample_id, left_path, right_path, gesture_label, gesture_id,
        scenario_id, scenario, is_negative
    """
    pos_path = Path(positives_csv or Paths.CADDY_CSV_POSITIVES)
    neg_path = Path(negatives_csv or Paths.CADDY_CSV_NEGATIVES)

    df_pos = pd.read_csv(pos_path)
    df_neg = pd.read_csv(neg_path)

    df_pos_real = df_pos[df_pos["synthetic"] == 0].copy()
    df_neg_real = df_neg[df_neg["synthetic"] == 0].copy()

    records = []

    # Process positive gesture samples (gesture_id: 0..15)
    for i, (_, row) in enumerate(df_pos_real.iterrows()):
        scenario_str = str(row["scenario"]).strip()
        scen_id = SCENARIO_TO_ID[scenario_str]
        left = str(row["stereo left"]).lstrip("/")
        right = str(row["stereo right"]).lstrip("/")
        label_name = str(row["label name"]).strip()
        label_id = int(row["label id"])
        sample_id = f"pos_{i:05d}"

        records.append(
            {
                "sample_id": sample_id,
                "left_path": left,
                "right_path": right,
                "gesture_label": label_name,
                "gesture_id": label_id,
                "scenario_id": scen_id,
                "scenario": scenario_str,
                "is_negative": 0,
            }
        )

    # Process negative gesture samples (gesture_id: 16, label: true_neg)
    for i, (_, row) in enumerate(df_neg_real.iterrows()):
        scenario_str = str(row["scenario"]).strip()
        scen_id = SCENARIO_TO_ID[scenario_str]
        left = str(row["stereo left"]).lstrip("/")
        right = str(row["stereo right"]).lstrip("/")
        label_name = "true_neg"
        label_id = 16
        sample_id = f"neg_{i:05d}"

        records.append(
            {
                "sample_id": sample_id,
                "left_path": left,
                "right_path": right,
                "gesture_label": label_name,
                "gesture_id": label_id,
                "scenario_id": scen_id,
                "scenario": scenario_str,
                "is_negative": 1,
            }
        )

    manifest = pd.DataFrame(records)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(out, index=False)

    return manifest


def make_standard_split(
    manifest: pd.DataFrame | None = None,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = SEED,
    output_dir: Path | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Standard train/validation/test split (70/15/15, stratified by gesture_id).

    Chained stratification:
      1. Split manifest into remainder (85%) and test (15%), stratified by gesture_id.
      2. Split remainder into train (70% total) and val (15% total), stratified by gesture_id.

    Saves:
      standard_train.csv, standard_val.csv, standard_test.csv under output_dir.
    """
    if manifest is None:
        manifest_path = Path("data/processed/manifest.csv")
        if not manifest_path.exists():
            manifest = build_manifest(output_path=manifest_path)
        else:
            manifest = pd.read_csv(manifest_path)

    # 1. Carve off test set (15% of total, stratified)
    remainder, df_test = train_test_split(
        manifest,
        test_size=test_size,
        stratify=manifest["gesture_id"],
        random_state=seed,
    )

    # 2. Carve off val set from remainder (val_size / (1 - test_size) fraction, stratified)
    val_relative = val_size / (1.0 - test_size)
    df_train, df_val = train_test_split(
        remainder,
        test_size=val_relative,
        stratify=remainder["gesture_id"],
        random_state=seed,
    )

    # Anti-leakage checks (pairwise intersections must be empty)
    s_train = set(df_train["sample_id"])
    s_val = set(df_val["sample_id"])
    s_test = set(df_test["sample_id"])

    assert len(s_train & s_val) == 0, f"Overlap between train and val: {len(s_train & s_val)}"
    assert len(s_train & s_test) == 0, f"Overlap between train and test: {len(s_train & s_test)}"
    assert len(s_val & s_test) == 0, f"Overlap between val and test: {len(s_val & s_test)}"

    out_dir = Path(output_dir or Paths.SPLITS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_train.to_csv(out_dir / "standard_train.csv", index=False)
    df_val.to_csv(out_dir / "standard_val.csv", index=False)
    df_test.to_csv(out_dir / "standard_test.csv", index=False)

    return df_train, df_val, df_test


def make_scenario_splits(
    manifest: pd.DataFrame | None = None,
    val_fraction: float = 0.15,
    seed: int = SEED,
    output_dir: Path | str | None = None,
) -> dict[int, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Scenario-aware Leave-One-Scenario-Out (LOSO) splits (8 folds).

    Per fold i (1..8):
      - Held-out scenario is the test set in full.
      - Pooled data from remaining 7 scenarios is stratified-split into train/val
        (val_fraction of the pooled subset).

    Saves:
      scenario_holdout_{i}_train.csv, scenario_holdout_{i}_val.csv, scenario_holdout_{i}_test.csv
      for i in 1..8 under output_dir.
    """
    if manifest is None:
        manifest_path = Path("data/processed/manifest.csv")
        if not manifest_path.exists():
            manifest = build_manifest(output_path=manifest_path)
        else:
            manifest = pd.read_csv(manifest_path)

    out_dir = Path(output_dir or Paths.SPLITS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    folds = {}

    for idx, scen_name in enumerate(SCENARIOS):
        fold_num = idx + 1  # 1-indexed (1..8)

        # Test set is the held-out scenario in full
        df_test = manifest[manifest["scenario"] == scen_name].copy()

        # Remaining 7 scenarios pooled
        remainder = manifest[manifest["scenario"] != scen_name].copy()

        # Stratified split of pooled remaining scenarios into train and val
        df_train, df_val = train_test_split(
            remainder,
            test_size=val_fraction,
            stratify=remainder["gesture_id"],
            random_state=seed,
        )

        # Programmatic verification checks
        assert set(df_test["scenario"]) == {
            scen_name
        }, f"Fold {fold_num}: test set contains scenarios other than {scen_name}"
        assert scen_name not in set(
            df_train["scenario"]
        ), f"Fold {fold_num}: train set contains held-out scenario {scen_name}"
        assert scen_name not in set(
            df_val["scenario"]
        ), f"Fold {fold_num}: val set contains held-out scenario {scen_name}"

        s_train = set(df_train["sample_id"])
        s_val = set(df_val["sample_id"])
        s_test = set(df_test["sample_id"])

        assert len(s_train & s_val) == 0, f"Fold {fold_num}: train ∩ val overlap"
        assert len(s_train & s_test) == 0, f"Fold {fold_num}: train ∩ test overlap"
        assert len(s_val & s_test) == 0, f"Fold {fold_num}: val ∩ test overlap"

        df_train.to_csv(out_dir / f"scenario_holdout_{fold_num}_train.csv", index=False)
        df_val.to_csv(out_dir / f"scenario_holdout_{fold_num}_val.csv", index=False)
        df_test.to_csv(out_dir / f"scenario_holdout_{fold_num}_test.csv", index=False)

        folds[fold_num] = (df_train, df_val, df_test)

    return folds


if __name__ == "__main__":
    out_file = Path("data/processed/manifest.csv")
    manifest_df = build_manifest(output_path=out_file)
    print(f"Manifest built successfully: {len(manifest_df):,} rows -> {out_file}")

    train_df, val_df, test_df = make_standard_split(manifest=manifest_df)
    print(
        f"Standard split built: Train={len(train_df):,} ({len(train_df)/len(manifest_df):.1%}), "
        f"Val={len(val_df):,} ({len(val_df)/len(manifest_df):.1%}), "
        f"Test={len(test_df):,} ({len(test_df)/len(manifest_df):.1%})"
    )

    loso_folds = make_scenario_splits(manifest=manifest_df)
    print(f"LOSO scenario splits built: 8 folds ({len(loso_folds)*3} CSV files saved).")
