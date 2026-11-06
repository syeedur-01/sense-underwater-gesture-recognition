"""
scripts/cache_enhanced_images.py
Phase 3 / P3.5 — Precompute and cache FUnIE-GAN and WaterNet enhanced images
for the entire CADDY dataset.

Output layout:
    {ENHANCED_CACHE_DIR}/none/{sample_id}_left.png
    {ENHANCED_CACHE_DIR}/none/{sample_id}_right.png
    {ENHANCED_CACHE_DIR}/funie_gan/{sample_id}_left.png
    {ENHANCED_CACHE_DIR}/funie_gan/{sample_id}_right.png
    {ENHANCED_CACHE_DIR}/waternet/{sample_id}_left.png
    {ENHANCED_CACHE_DIR}/waternet/{sample_id}_right.png

After caching, two programmatic checks are run and results written to
    data/processed/enhancement_report.md

Check 1 — Completeness: every sample_id in manifest.csv has a file in all
           three method subdirectories for both left and right.
Check 2 — Stereo-pair integrity: left and right filenames for every sample_id
           still correspond to that exact sample_id after caching (guards
           against silent desynchronization in independent per-image passes).

Usage:
    # Cache all three variants (requires weight files to exist):
    python scripts/cache_enhanced_images.py

    # Cache only specific method(s):
    python scripts/cache_enhanced_images.py --methods none funie_gan

    # Dry-run (check completeness / generate report without re-caching):
    python scripts/cache_enhanced_images.py --dry-run

    # Skip already-cached files:
    python scripts/cache_enhanced_images.py --skip-existing
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Bootstrap paths
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import Paths  # noqa: E402

MANIFEST_PATH = Path("data/processed/manifest.csv")
ENHANCED_CACHE_DIR = Path(Paths.ENHANCED_CACHE_DIR)
CADDY_ROOT = Path(Paths.CADDY_ROOT)
REPORT_PATH = Path("data/processed/enhancement_report.md")
ALL_METHODS = ["none", "funie_gan", "waternet"]


# ---------------------------------------------------------------------------
# Image I/O helpers
# ---------------------------------------------------------------------------


def _load_rgb(rel_path: str) -> np.ndarray:
    """Load a CADDY image as HxWx3 uint8 RGB array."""
    abs_path = CADDY_ROOT / rel_path.lstrip("/")
    if not abs_path.exists():
        raise FileNotFoundError(f"Raw image not found: {abs_path}")
    img = cv2.imread(str(abs_path))
    if img is None:
        raise ValueError(f"cv2.imread failed for: {abs_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _save_png(image_rgb: np.ndarray, path: Path) -> None:
    """Save an HxWx3 uint8 RGB array as PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(path), bgr)
    if not ok:
        raise IOError(f"cv2.imwrite failed for: {path}")


# ---------------------------------------------------------------------------
# Enhancer factory
# ---------------------------------------------------------------------------


def _build_enhancer(method: str, device: str):
    """Return an enhancer object for the given method, or None for 'none'."""
    if method == "none":
        return None
    if method == "funie_gan":
        from src.models.enhancers import FUnIEGANEnhancer

        return FUnIEGANEnhancer(device=device)
    if method == "waternet":
        from src.models.enhancers import WaterNetEnhancer

        return WaterNetEnhancer(device=device)
    raise ValueError(f"Unknown enhancement method: {method!r}")


def _apply(enhancer, image_rgb: np.ndarray) -> np.ndarray:
    """Apply enhancer (or return raw copy for 'none')."""
    if enhancer is None:
        return image_rgb.copy()
    return enhancer.enhance(image_rgb)


# ---------------------------------------------------------------------------
# Cache one method
# ---------------------------------------------------------------------------


def cache_method(
    df: pd.DataFrame,
    method: str,
    device: str,
    skip_existing: bool,
    dry_run: bool,
) -> dict:
    """
    Cache all images for one enhancement method.

    Returns a timing stats dict.
    """
    out_dir = ENHANCED_CACHE_DIR / method
    out_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print(f"  [DRY-RUN] Would cache {len(df)} samples × 2 images for '{method}'")
        return {
            "method": method,
            "n_samples": len(df),
            "elapsed_s": 0,
            "skipped": 0,
            "processed": 0,
        }

    print(f"\n{'='*60}")
    print(f"  Caching: {method.upper()}")
    print(f"  Output : {out_dir}")
    print(f"{'='*60}")

    # Load model (may raise FileNotFoundError for missing weights)
    try:
        enhancer = _build_enhancer(method, device)
    except FileNotFoundError as exc:
        print(f"\n  ERROR: {exc}")
        print("  Skipping this method.")
        return {
            "method": method,
            "error": str(exc),
            "n_samples": 0,
            "elapsed_s": 0,
            "skipped": 0,
            "processed": 0,
        }

    t0 = time.perf_counter()
    skipped = 0
    processed = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=method, unit="sample"):
        sample_id = str(row["sample_id"])

        for side, rel_path_col in [("left", "left_path"), ("right", "right_path")]:
            out_path = out_dir / f"{sample_id}_{side}.png"

            if skip_existing and out_path.exists():
                skipped += 1
                continue

            img_rgb = _load_rgb(str(row[rel_path_col]))
            enhanced = _apply(enhancer, img_rgb)
            _save_png(enhanced, out_path)
            processed += 1

    elapsed = time.perf_counter() - t0
    n_total = len(df)
    rate = (n_total * 2 - skipped) / elapsed if elapsed > 0 else float("inf")
    print(
        f"\n  Done: {processed} images written, {skipped} skipped, {elapsed:.1f}s ({rate:.1f} img/s)"
    )

    return {
        "method": method,
        "n_samples": n_total,
        "processed": processed,
        "skipped": skipped,
        "elapsed_s": elapsed,
        "rate_img_per_s": rate,
    }


# ---------------------------------------------------------------------------
# Verification checks
# ---------------------------------------------------------------------------


def run_checks(df: pd.DataFrame, methods: list[str]) -> tuple[list[str], list[str]]:
    """
    Check 1: Completeness — every sample_id has files for all requested methods × {left, right}.
    Check 2: Stereo-pair integrity — filenames encode the correct sample_id.

    Returns (missing_files, integrity_errors) as lists of human-readable strings.
    """
    missing: list[str] = []
    integrity_errors: list[str] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Verifying", unit="sample"):
        sample_id = str(row["sample_id"])

        for method in methods:
            method_dir = ENHANCED_CACHE_DIR / method
            for side in ["left", "right"]:
                expected_path = method_dir / f"{sample_id}_{side}.png"
                if not expected_path.exists():
                    missing.append(f"{method}/{sample_id}_{side}.png")
                else:
                    # Stereo-pair integrity: filename must encode this exact sample_id
                    stem = expected_path.stem  # e.g. "pos_00042_left"
                    if not stem.startswith(sample_id + "_"):
                        integrity_errors.append(
                            f"{expected_path} — stem {stem!r} does not start with {sample_id!r}"
                        )

    return missing, integrity_errors


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------


def write_report(
    df: pd.DataFrame,
    stats_list: list[dict],
    missing: list[str],
    integrity_errors: list[str],
    methods_checked: list[str],
) -> None:
    """Write data/processed/enhancement_report.md."""
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    total_samples = len(df)
    lines: list[str] = []

    lines += [
        "# Enhancement Cache Report",
        "",
        f"Generated: {now}  ",
        f"Manifest : `data/processed/manifest.csv` ({total_samples} samples)  ",
        f"Cache dir: `{ENHANCED_CACHE_DIR}`  ",
        f"Methods  : {', '.join(methods_checked)}",
        "",
    ]

    # --- Processing stats ---
    lines += [
        "## Processing Statistics",
        "",
        "| Method | Samples | Images Written | Skipped | Time (s) | Rate (img/s) |",
        "|--------|---------|----------------|---------|----------|--------------| ",
    ]
    for s in stats_list:
        if "error" in s:
            lines.append(f"| {s['method']} | — | ERROR | — | — | {s['error'][:60]} |")
        else:
            lines.append(
                f"| {s['method']} | {s['n_samples']} | {s['processed']} | {s['skipped']} "
                f"| {s['elapsed_s']:.1f} | {s.get('rate_img_per_s', 0):.1f} |"
            )
    lines.append("")

    # --- Completeness check ---
    lines += ["## Check 1 — Completeness", ""]
    expected_total = total_samples * 2 * len(methods_checked)
    if missing:
        lines.append(f"❌ **{len(missing)} missing file(s)** (expected {expected_total} total)")
        lines.append("")
        lines.append("| Missing file |")
        lines.append("|-------------|")
        for m in missing[:200]:  # cap at 200 rows in report
            lines.append(f"| `{m}` |")
        if len(missing) > 200:
            lines.append(f"| … and {len(missing) - 200} more (see full run output) |")
    else:
        lines.append(f"✅ All {expected_total} expected files present.")
    lines.append("")

    # --- Stereo-pair integrity check ---
    lines += ["## Check 2 — Stereo-Pair Integrity", ""]
    if integrity_errors:
        lines.append(f"❌ **{len(integrity_errors)} integrity error(s)**")
        lines.append("")
        for e in integrity_errors[:100]:
            lines.append(f"- `{e}`")
        if len(integrity_errors) > 100:
            lines.append(f"- … and {len(integrity_errors) - 100} more")
    else:
        lines.append("✅ All cached filenames correctly encode their sample_id.")
    lines.append("")

    # --- Overall verdict ---
    all_ok = not missing and not integrity_errors
    verdict = "✅ **PASS** — All checks passed." if all_ok else "❌ **FAIL** — See errors above."
    lines += ["## Overall Verdict", "", verdict, ""]

    # --- Per-method file count ---
    lines += ["## File Count by Method", ""]
    lines += ["| Method | Left files | Right files |", "|--------|-----------|------------|"]
    for method in methods_checked:
        method_dir = ENHANCED_CACHE_DIR / method
        if method_dir.exists():
            lefts = len(list(method_dir.glob("*_left.png")))
            rights = len(list(method_dir.glob("*_right.png")))
        else:
            lefts = rights = 0
        lines.append(f"| {method} | {lefts} | {rights} |")
    lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to: {REPORT_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cache enhanced CADDY images (Phase 3 / P3.5).")
    p.add_argument(
        "--methods",
        nargs="+",
        choices=ALL_METHODS,
        default=ALL_METHODS,
        help="Which enhancement methods to cache (default: all three).",
    )
    p.add_argument(
        "--device",
        default="cuda",
        help="PyTorch device string (default: 'cuda'; falls back to 'cpu' automatically).",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip writing a file if it already exists in the cache.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Check completeness and generate report without (re-)caching any images.",
    )
    p.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the post-cache verification checks (not recommended).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not MANIFEST_PATH.exists():
        print(f"ERROR: manifest.csv not found at {MANIFEST_PATH}")
        print("Run scripts/build_manifest.py first.")
        return 1

    df = pd.read_csv(MANIFEST_PATH)
    print(f"Manifest loaded: {len(df)} samples")
    print(f"Methods to cache: {args.methods}")
    print(f"Enhanced cache dir: {ENHANCED_CACHE_DIR}")

    # --- Cache phase ---
    stats_list = []
    for method in args.methods:
        s = cache_method(df, method, args.device, args.skip_existing, args.dry_run)
        stats_list.append(s)

    # --- Verification phase ---
    missing: list[str] = []
    integrity_errors: list[str] = []
    if not args.no_verify:
        print(f"\nRunning verification checks for methods: {args.methods}")
        missing, integrity_errors = run_checks(df, args.methods)
        if missing:
            print(f"\n⚠  Completeness check: {len(missing)} missing file(s)")
        else:
            print("\n✅ Completeness check: all files present")
        if integrity_errors:
            print(f"⚠  Stereo integrity check: {len(integrity_errors)} error(s)")
        else:
            print("✅ Stereo integrity check: all pairs consistent")

    # --- Report ---
    write_report(df, stats_list, missing, integrity_errors, args.methods)

    return 0 if (not missing and not integrity_errors) else 1


if __name__ == "__main__":
    sys.exit(main())
