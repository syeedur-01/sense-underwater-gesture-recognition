"""
scripts/generate_splits.py
Generates all data manifests and split CSVs from the raw CADDY annotations.

Run this after downloading the CADDY dataset and configuring .env with the
correct CADDY_ROOT, CADDY_CSV_POSITIVES, and CADDY_CSV_NEGATIVES paths.

Outputs written to data/processed/:
    manifest.csv                        — full canonical manifest (16,429 rows)
    splits/standard_train.csv           — standard 70% training split
    splits/standard_val.csv             — standard 15% validation split
    splits/standard_test.csv            — standard 15% test split
    splits/scenario_holdout_1_train.csv — LOSO fold 1 (held-out: biograd-A)
    splits/scenario_holdout_1_val.csv
    splits/scenario_holdout_1_test.csv
    ... (8 folds total = 24 scenario-split CSVs)

Usage:
    python scripts/generate_splits.py
"""

import io
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import SEED, Paths  # noqa: E402
from src.data.splits import (  # noqa: E402
    SCENARIOS,
    build_manifest,
    make_scenario_splits,
    make_standard_split,
)

MANIFEST_PATH = Path("data/processed/manifest.csv")
SPLITS_DIR = Path(Paths.SPLITS_DIR)


def main() -> None:
    print("=" * 60)
    print("SENSE — Generate Data Manifests and Splits")
    print(f"SEED = {SEED}")
    print("=" * 60)

    # ----------------------------------------------------------------
    # 1. Build canonical manifest
    # ----------------------------------------------------------------
    print("\n[1/3] Building canonical manifest ...")
    manifest = build_manifest(output_path=MANIFEST_PATH)
    n_pos = int((manifest["is_negative"] == 0).sum())
    n_neg = int((manifest["is_negative"] == 1).sum())
    print(f"      Manifest: {len(manifest):,} rows  ({n_pos:,} positives + {n_neg:,} negatives)")
    print(f"      Saved to: {MANIFEST_PATH}")

    # ----------------------------------------------------------------
    # 2. Standard 70/15/15 stratified split
    # ----------------------------------------------------------------
    print("\n[2/3] Building standard 70/15/15 stratified split ...")
    train_df, val_df, test_df = make_standard_split(manifest=manifest, output_dir=SPLITS_DIR)
    pct = lambda df: f"{len(df)/len(manifest):.1%}"  # noqa: E731
    print(f"      Train : {len(train_df):>6,} rows  ({pct(train_df)})")
    print(f"      Val   : {len(val_df):>6,} rows  ({pct(val_df)})")
    print(f"      Test  : {len(test_df):>6,} rows  ({pct(test_df)})")
    print(f"      Saved to: {SPLITS_DIR}/standard_{{train,val,test}}.csv")

    # ----------------------------------------------------------------
    # 3. Scenario LOSO splits (8 folds)
    # ----------------------------------------------------------------
    print("\n[3/3] Building scenario LOSO splits (8 folds) ...")
    folds = make_scenario_splits(manifest=manifest, output_dir=SPLITS_DIR)
    for fold_num, (tr, va, te) in folds.items():
        held_out = SCENARIOS[fold_num - 1]
        print(
            f"      Fold {fold_num} | held-out={held_out:<15} | "
            f"test={len(te):>4} | train={len(tr):>5} | val={len(va):>4}"
        )
    print(f"      Saved to: {SPLITS_DIR}/scenario_holdout_{{1..8}}_{{train,val,test}}.csv")

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    csv_files = list(SPLITS_DIR.glob("*.csv")) + [MANIFEST_PATH]
    print()
    print("=" * 60)
    print(f"Done. Generated {len(csv_files)} CSV files total:")
    print(f"  {MANIFEST_PATH}")
    for f in sorted(SPLITS_DIR.glob("*.csv")):
        print(f"  {f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
