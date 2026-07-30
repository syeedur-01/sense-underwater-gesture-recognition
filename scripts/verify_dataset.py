"""
scripts/verify_dataset.py
P1.2 + P1.3 — Dataset extraction inventory and CSV annotation validation.

CADDY Dataset Schema (confirmed from CADDY_gestures_complete_v2_release):
  Positives CSV columns: index, scenario, 'stereo left', 'stereo right',
                         'label name', 'label id', roi left, roi right,
                         synthetic, iqa_mdm_entropy, iqa_mdm_d, iqa_mdm_dcomp,
                         distortion, 'param 1', 'param 2'
  Negatives CSV columns: index, scenario, 'stereo left', 'stereo right',
                         'selected img', 'label name', 'label id', 'roi params',
                         synthetic, 'distortion type', 'dist. param 1', 'dist. param 2'
  Image paths in CSV are relative to CADDY_ROOT (start with /{scenario}/...)
  Naming convention: {scenario}_{nnnnn}_left.jpg / {scenario}_{nnnnn}_right.jpg

Usage:
    python scripts/verify_dataset.py

Outputs:
    Console report (human-readable)
    data/processed/dataset_audit_report.txt (persisted copy)

Phase 1 exit criterion: zero "missing file" errors, 8 scenarios, 16 gesture
classes (15 positive + no-gesture negative).
"""

import io
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows (cp1252 terminals block checkmark chars)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd

# ---------------------------------------------------------------------------
# Bootstrap paths
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import Paths  # noqa: E402

CADDY_ROOT = Path(Paths.CADDY_ROOT)
CSV_POSITIVES = Path(Paths.CADDY_CSV_POSITIVES)
CSV_NEGATIVES = Path(Paths.CADDY_CSV_NEGATIVES)
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_REPORT_PATH = PROCESSED_DIR / "dataset_audit_report.txt"

# Expected values per CADDY paper + Final_plan.md
EXPECTED_SCENARIOS = 8
EXPECTED_GESTURE_CLASSES = 16  # 15 positive gestures + 1 no-gesture class = 16 total
EXPECTED_POS_REAL_MIN = 8500  # allow ±~8% of 9,239 stated in paper
EXPECTED_POS_REAL_MAX = 10000
EXPECTED_NEG_REAL_MIN = 6500  # allow ±~10% of 7,190 stated in paper
EXPECTED_NEG_REAL_MAX = 8000

SCENARIO_NAMES = [
    "biograd-A",
    "biograd-B",
    "biograd-C",
    "brodarski-A",
    "brodarski-B",
    "brodarski-C",
    "brodarski-D",
    "genova-A",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def hr(char="=", width=72) -> str:
    return char * width


def rel(path: Path) -> str:
    """Return path relative to CWD for cleaner output."""
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------
def run_audit() -> tuple[list[str], list[str]]:
    lines: list[str] = []
    errors: list[str] = []

    def log(msg: str = "") -> None:
        lines.append(msg)
        print(msg)

    log(hr())
    log("CADDY DATASET AUDIT REPORT")
    log(hr())
    log(f"CADDY_ROOT       : {CADDY_ROOT}")
    log(f"CSV_POSITIVES    : {rel(CSV_POSITIVES)}")
    log(f"CSV_NEGATIVES    : {rel(CSV_NEGATIVES)}")
    log()

    # ------------------------------------------------------------------
    # SECTION 1: Verify CADDY_ROOT and CSVs exist
    # ------------------------------------------------------------------
    log(hr("-"))
    log("SECTION 1 — Root & CSV File Existence")
    log(hr("-"))

    if not CADDY_ROOT.exists():
        msg = f"ERROR: CADDY_ROOT does not exist: {CADDY_ROOT}"
        log(f"  *** {msg}")
        errors.append(msg)
        log()
        return lines, errors
    log(f"  ✓ CADDY_ROOT exists: {CADDY_ROOT}")

    for name, path in [("Positives CSV", CSV_POSITIVES), ("Negatives CSV", CSV_NEGATIVES)]:
        if not path.exists():
            msg = f"ERROR: {name} not found: {path}"
            log(f"  *** {msg}")
            errors.append(msg)
        else:
            size_mb = path.stat().st_size / 1e6
            log(f"  ✓ {name} exists ({size_mb:.1f} MB): {rel(path)}")
    log()

    if errors:
        return lines, errors

    # ------------------------------------------------------------------
    # SECTION 2: Scenario folder structure
    # ------------------------------------------------------------------
    log(hr("-"))
    log("SECTION 2 — Scenario Folder Structure")
    log(hr("-"))

    found_scenarios = sorted(
        [
            d.name
            for d in CADDY_ROOT.iterdir()
            if d.is_dir() and not d.name.startswith(".") and not d.name.endswith(".zip")
        ]
    )
    log(f"Scenario folders found ({len(found_scenarios)}):")
    for s in found_scenarios:
        log(f"  {s}")
    log()

    if len(found_scenarios) != EXPECTED_SCENARIOS:
        msg = (
            f"ERROR: Expected {EXPECTED_SCENARIOS} scenario folders, "
            f"found {len(found_scenarios)}"
        )
        log(f"  *** {msg}")
        errors.append(msg)
    else:
        log(f"  ✓ Exactly {EXPECTED_SCENARIOS} scenario folders confirmed.")

    missing_scenarios = set(SCENARIO_NAMES) - set(found_scenarios)
    if missing_scenarios:
        msg = f"ERROR: Missing expected scenario folders: {sorted(missing_scenarios)}"
        log(f"  *** {msg}")
        errors.append(msg)
    log()

    # Check left/right subfolder naming convention (sample one scenario)
    sample_scenario = CADDY_ROOT / "biograd-A"
    log("Subfolder structure (biograd-A):")
    for entry in sorted(sample_scenario.rglob("*")):
        if entry.is_dir():
            log(f"  {rel(entry)}")
    log()

    # Confirm naming convention with sample images
    sample_images = list((sample_scenario / "true_positives" / "raw").glob("*.jpg"))[:4]
    log("Sample image filenames:")
    for img in sample_images:
        log(f"  {img.name}")
    log()

    # ------------------------------------------------------------------
    # SECTION 3: Image inventory
    # ------------------------------------------------------------------
    log(hr("-"))
    log("SECTION 3 — Image File Count")
    log(hr("-"))

    total_images = 0
    ext_counts: dict[str, int] = {}
    for p in CADDY_ROOT.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            total_images += 1
            ext_counts[p.suffix.lower()] = ext_counts.get(p.suffix.lower(), 0) + 1

    log(f"Total image files (all types incl. synthetic): {total_images}")
    for ext, cnt in sorted(ext_counts.items()):
        log(f"  {ext}: {cnt:,}")
    log()

    # ------------------------------------------------------------------
    # SECTION 4: Positives CSV validation
    # ------------------------------------------------------------------
    log(hr("-"))
    log("SECTION 4 — Positives CSV Validation")
    log(hr("-"))

    df_pos = pd.read_csv(CSV_POSITIVES)
    log(f"Total rows (incl. synthetic): {len(df_pos):,}")
    log(f"Columns: {list(df_pos.columns)}")
    log()

    # Real-only (non-synthetic)
    df_pos_real = df_pos[df_pos["synthetic"] == 0].copy()
    log(f"Real samples (synthetic==0): {len(df_pos_real):,}")
    log(f"Synthetic samples: {len(df_pos) - len(df_pos_real):,}")
    log()

    if not (EXPECTED_POS_REAL_MIN <= len(df_pos_real) <= EXPECTED_POS_REAL_MAX):
        msg = (
            f"WARNING: Real positive samples ({len(df_pos_real)}) outside expected range "
            f"[{EXPECTED_POS_REAL_MIN}, {EXPECTED_POS_REAL_MAX}]."
        )
        log(f"  *** {msg}")
        errors.append(msg)
    else:
        log(f"  ✓ Real positive count {len(df_pos_real):,} within expected range.")

    # Gesture classes
    gesture_col = "label name"
    label_id_col = "label id"
    gestures = df_pos_real[[gesture_col, label_id_col]].drop_duplicates().sort_values(label_id_col)
    log(f"\nGesture classes in real positives ({len(gestures)}):")
    for _, row in gestures.iterrows():
        log(f"  [{int(row[label_id_col]):2d}] {row[gesture_col]}")
    log()

    num_gesture_classes = len(gestures)
    if num_gesture_classes != EXPECTED_GESTURE_CLASSES:
        msg = (
            f"WARNING: Expected {EXPECTED_GESTURE_CLASSES} positive gesture classes, "
            f"found {num_gesture_classes}"
        )
        log(f"  *** {msg}")
        errors.append(msg)
    else:
        log(f"  ✓ Exactly {EXPECTED_GESTURE_CLASSES} gesture classes confirmed.")
    log()

    # Missing value check
    null_counts = (
        df_pos_real[[gesture_col, label_id_col, "scenario", "stereo left", "stereo right"]]
        .isnull()
        .sum()
    )
    if null_counts.any():
        bad = null_counts[null_counts > 0].to_dict()
        msg = f"ERROR: Nulls found in positives CSV critical columns: {bad}"
        log(f"  *** {msg}")
        errors.append(msg)
    else:
        log("  ✓ No nulls in critical columns (gesture, label_id, scenario, stereo paths).")

    # Scenario distribution
    log("\nReal positive samples per scenario:")
    for scenario, count in df_pos_real["scenario"].value_counts().sort_index().items():
        log(f"  {scenario}: {count:,}")
    log()

    # ------------------------------------------------------------------
    # SECTION 5: Negatives CSV validation
    # ------------------------------------------------------------------
    log(hr("-"))
    log("SECTION 5 — Negatives CSV Validation")
    log(hr("-"))

    df_neg = pd.read_csv(CSV_NEGATIVES)
    log(f"Total rows (incl. synthetic): {len(df_neg):,}")
    log(f"Columns: {list(df_neg.columns)}")
    log()

    df_neg_real = df_neg[df_neg["synthetic"] == 0].copy()
    log(f"Real samples (synthetic==0): {len(df_neg_real):,}")
    log(f"Synthetic samples: {len(df_neg) - len(df_neg_real):,}")
    log()

    if not (EXPECTED_NEG_REAL_MIN <= len(df_neg_real) <= EXPECTED_NEG_REAL_MAX):
        msg = (
            f"WARNING: Real negative samples ({len(df_neg_real)}) outside expected range "
            f"[{EXPECTED_NEG_REAL_MIN}, {EXPECTED_NEG_REAL_MAX}]."
        )
        log(f"  *** {msg}")
        errors.append(msg)
    else:
        log(f"  ✓ Real negative count {len(df_neg_real):,} within expected range.")

    neg_label_ids = df_neg_real["label id"].unique()
    if set(neg_label_ids) != {-1}:
        msg = f"WARNING: Unexpected label IDs in negatives: {neg_label_ids}"
        log(f"  *** {msg}")
        errors.append(msg)
    else:
        log("  ✓ All negative samples have label_id == -1 (no-gesture).")

    log("\nReal negative samples per scenario:")
    for scenario, count in df_neg_real["scenario"].value_counts().sort_index().items():
        log(f"  {scenario}: {count:,}")
    log()

    # ------------------------------------------------------------------
    # SECTION 6: File existence check (real images only)
    # ------------------------------------------------------------------
    log(hr("-"))
    log("SECTION 6 — Referenced File Existence Check (real images only)")
    log(hr("-"))

    # Build disk lookup: path relative to CADDY_ROOT → absolute path
    disk_paths: set[str] = set()
    for p in CADDY_ROOT.rglob("*.jpg"):
        # Store as relative path string matching CSV format: /scenario/...
        try:
            rel_path = "/" + p.relative_to(CADDY_ROOT).as_posix()
            disk_paths.add(rel_path)
        except ValueError:
            pass

    log(f"Total .jpg files indexed from disk: {len(disk_paths):,}")

    # Check positives
    missing_pos = 0
    missing_pos_examples: list[str] = []
    for ref in df_pos_real["stereo left"]:
        ref_str = str(ref).strip()
        if ref_str not in disk_paths:
            missing_pos += 1
            if len(missing_pos_examples) < 5:
                missing_pos_examples.append(ref_str)

    log(f"\nPositives — missing left images: {missing_pos:,} / {len(df_pos_real):,}")
    if missing_pos > 0:
        log("  First missing examples:")
        for ex in missing_pos_examples:
            log(f"    {ex}")
        msg = f"ERROR: {missing_pos} positive left image references not found on disk."
        log(f"  *** {msg}")
        errors.append(msg)
    else:
        log("  ✓ All positive left image references found on disk.")

    # Check negatives
    missing_neg = 0
    missing_neg_examples: list[str] = []
    for ref in df_neg_real["stereo left"]:
        ref_str = str(ref).strip()
        if ref_str not in disk_paths:
            missing_neg += 1
            if len(missing_neg_examples) < 5:
                missing_neg_examples.append(ref_str)

    log(f"\nNegatives — missing left images: {missing_neg:,} / {len(df_neg_real):,}")
    if missing_neg > 0:
        log("  First missing examples:")
        for ex in missing_neg_examples:
            log(f"    {ex}")
        msg = f"ERROR: {missing_neg} negative left image references not found on disk."
        log(f"  *** {msg}")
        errors.append(msg)
    else:
        log("  ✓ All negative left image references found on disk.")
    log()

    # ------------------------------------------------------------------
    # SECTION 7: Summary
    # ------------------------------------------------------------------
    log(hr())
    log("AUDIT SUMMARY")
    log(hr())
    log(f"  Total real positive stereo pairs : {len(df_pos_real):,}")
    log(f"  Total real negative stereo pairs : {len(df_neg_real):,}")
    log(f"  Total real stereo pairs combined : {len(df_pos_real) + len(df_neg_real):,}")
    log(f"  Scenario folders                 : {len(found_scenarios)}")
    log(f"  Gesture classes (positive)       : {num_gesture_classes}")
    log("  No-gesture class                 : 1 (label_id=-1, negatives CSV)")
    log(f"  Total classes (model output)     : {num_gesture_classes + 1}")
    log()

    if errors:
        log(f"RESULT: {len(errors)} ISSUE(S) FOUND:")
        for e in errors:
            log(f"  ✗ {e}")
    else:
        log("RESULT: ✓ ALL CHECKS PASSED — dataset is clean and ready for Phase 2.")
    log(hr())

    return lines, errors


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    lines, errors = run_audit()

    with open(AUDIT_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nAudit report saved to: {AUDIT_REPORT_PATH}")

    sys.exit(0 if not errors else 1)
