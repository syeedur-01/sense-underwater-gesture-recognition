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

from src.config import Paths  # noqa: E402

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


if __name__ == "__main__":
    out_file = Path("data/processed/manifest.csv")
    df = build_manifest(output_path=out_file)
    print(f"Manifest built successfully: {len(df):,} rows -> {out_file}")
