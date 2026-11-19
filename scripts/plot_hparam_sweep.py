"""
scripts/plot_hparam_sweep.py
Phase 7 — P7.3: build the sweep's two evidence artifacts.

Reads the metrics.csv (and best.pth, where present) of every recorded variation
and writes:

    results/hparam_sweep/comparison_plot.png
    results/hparam_sweep/summary_table.csv

Both artifacts are derived purely from already-completed runs — no training or
evaluation happens here, and the winner is not re-derived: it is whatever the
run-level selection rule documented in `data/processed/hparam_sweep_notes.md`
already named.

Usage (from project root, with venv active):
    python scripts/plot_hparam_sweep.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import matplotlib

matplotlib.use("Agg")  # headless — no GUI needed
import matplotlib.pyplot as plt
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "results" / "hparam_sweep"

plt.rcParams.update(
    {
        "figure.dpi": 100,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# Phase boundaries are fixed by configs/base.yaml: 1-10 / 11-30 / 31-50.
PHASES = [
    ("Phase 1 (frozen backbone)", 1, 10),
    ("Phase 2 (last 2 blocks)", 11, 30),
    ("Phase 3 (full backbone)", 31, 50),
]

# Every variation recorded in data/processed/hparam_sweep_notes.md.
# `diverged` runs are kept in the figure on purpose: they are the evidence that
# base_lr 1e-3 is outside the pipeline's stability region.
RUNS = [
    {
        "label": "Run 1 — sense_full (lr 1e-4)",
        "experiment": "sense_full",
        "run_dir": "runs/sense_full/20261114T122905Z",
        "status": "completed",
        "color": "#7f7f7f",
        "linestyle": "-",
        "linewidth": 1.4,
        "note": "reference configuration (base_lr 1e-4)",
    },
    {
        "label": "V1-A1 — lr 1e-3 (diverged)",
        "experiment": "sense_full_lr1e-3",
        "run_dir": "runs/sense_full_lr1e-3/20261115T142011Z",
        "status": "diverged",
        "color": "#d62728",
        "linestyle": "--",
        "linewidth": 1.2,
        "note": "diverged at epoch 6 (fp16 forward overflow); best state is epoch 5",
    },
    {
        "label": "V1-A2 — lr 1e-3 (guard abort)",
        "experiment": "sense_full_lr1e-3",
        "run_dir": "runs/sense_full_lr1e-3/20261117T162825Z",
        "status": "diverged",
        "color": "#ff9896",
        "linestyle": "--",
        "linewidth": 1.2,
        "note": "diverged; divergence guard aborted at epoch 5, batch 243",
    },
    {
        "label": "V1-A3 — lr 3e-4",
        "experiment": "sense_full_lr3e-4",
        "run_dir": "runs/sense_full_lr3e-4/20261117T172308Z",
        "status": "completed",
        "color": "#1f77b4",
        "linestyle": "-",
        "linewidth": 1.4,
        "note": "intermediate base LR; winner of variation 1",
    },
    {
        "label": "V2 — phase-3 lr_scale 0.1",
        "experiment": "sense_full_lr3e-4_p3lr0.1",
        "run_dir": "runs/sense_full_lr3e-4_p3lr0.1/20261119T063459Z",
        "status": "completed",
        "color": "#2ca02c",
        "linestyle": "-",
        "linewidth": 1.4,
        "note": "phase-3 backbone lr_scale 0.01 -> 0.1",
    },
    {
        "label": "V3 — phase-3 lr_scale 0.3 (winner)",
        "experiment": "sense_full_lr3e-4_p3lr0.3",
        "run_dir": "runs/sense_full_lr3e-4_p3lr0.3/20261119T120231Z",
        "status": "completed",
        "color": "#ff7f0e",
        "linestyle": "-",
        "linewidth": 2.4,
        "note": "phase-3 backbone lr_scale 0.1 -> 0.3; sweep winner",
    },
]


def _read_metrics(run_dir: Path) -> pd.DataFrame:
    """Load one run's per-epoch metrics.csv."""
    df = pd.read_csv(run_dir / "metrics.csv")
    for col in ("epoch", "val_macro_f1", "val_accuracy", "val_min_class_accuracy"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _read_wall_time(run_dir: Path) -> float | None:
    """Total wall time in seconds, if the tracker wrote runtime.txt."""
    path = run_dir / "runtime.txt"
    if not path.exists():
        return None
    try:
        # runtime.txt is written as e.g. "11411.95s"
        return float(path.read_text(encoding="utf-8").strip().rstrip("sS"))
    except ValueError:
        return None


def _best_from_checkpoint(run_dir: Path) -> dict | None:
    """Read the saved best checkpoint's own metrics, if one exists."""
    path = run_dir / "checkpoints" / "best.pth"
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    metrics = ckpt.get("metrics") or {}
    if not metrics:
        return None
    return {
        "epoch": int(ckpt.get("epoch", -1)),
        "loss": float(metrics.get("loss", float("nan"))),
        "accuracy": float(metrics.get("accuracy", float("nan"))),
        "macro_f1": float(metrics.get("macro_f1", float("nan"))),
        "min_class_accuracy": float(metrics.get("min_class_accuracy", float("nan"))),
    }


def _phase_mean(df: pd.DataFrame, lo: int, hi: int) -> float | None:
    """Mean val_macro_f1 over an epoch range, or None if the run never reached it."""
    window = df[(df["epoch"] >= lo) & (df["epoch"] <= hi)]
    if window.empty:
        return None
    return float(window["val_macro_f1"].mean())


def _hms(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    total = int(round(seconds))
    return f"{total // 3600}h{total % 3600 // 60:02d}m{total % 60:02d}s"


def build_rows() -> pd.DataFrame:
    rows = []
    for run in RUNS:
        run_dir = ROOT / run["run_dir"]
        df = _read_metrics(run_dir)
        best = _best_from_checkpoint(run_dir)
        if best is None:
            # Fall back to the metrics.csv peak for runs without a usable checkpoint.
            peak = df.loc[df["val_macro_f1"].idxmax()]
            best = {
                "epoch": int(peak["epoch"]),
                "loss": float(peak["val_loss"]),
                "accuracy": float(peak["val_accuracy"]),
                "macro_f1": float(peak["val_macro_f1"]),
                "min_class_accuracy": float(peak["val_min_class_accuracy"]),
            }
        wall = _read_wall_time(run_dir)
        p1, p2, p3 = (_phase_mean(df, lo, hi) for _, lo, hi in PHASES)
        rows.append(
            {
                "experiment": run["experiment"],
                "config": f"configs/{run['experiment']}.yaml",
                "run_dir": run["run_dir"],
                "status": run["status"],
                "best_epoch": best["epoch"],
                "best_val_macro_f1": round(best["macro_f1"], 6),
                "best_val_accuracy": round(best["accuracy"], 6),
                "best_val_min_class_accuracy": round(best["min_class_accuracy"], 6),
                "best_val_loss": round(best["loss"], 6),
                "phase1_mean_macro_f1": None if p1 is None else round(p1, 6),
                "phase2_mean_macro_f1": None if p2 is None else round(p2, 6),
                "phase3_mean_macro_f1": None if p3 is None else round(p3, 6),
                "epochs_logged": int(len(df)),
                "wall_time_s": None if wall is None else round(wall, 2),
                "wall_time_hms": _hms(wall),
                "note": run["note"],
            }
        )

    table = pd.DataFrame(rows)

    # Rank completed runs only — a diverged run's peak is not a comparable result.
    table["rank"] = None
    completed = table[table["status"] == "completed"].sort_values(
        "best_val_macro_f1", ascending=False
    )
    for position, index in enumerate(completed.index, start=1):
        table.loc[index, "rank"] = position

    ordered = table.sort_values(
        by=["rank", "best_val_macro_f1"],
        ascending=[True, False],
        na_position="last",
    )
    columns = [
        "rank",
        "experiment",
        "config",
        "run_dir",
        "status",
        "best_epoch",
        "best_val_macro_f1",
        "best_val_accuracy",
        "best_val_min_class_accuracy",
        "best_val_loss",
        "phase1_mean_macro_f1",
        "phase2_mean_macro_f1",
        "phase3_mean_macro_f1",
        "epochs_logged",
        "wall_time_s",
        "wall_time_hms",
        "note",
    ]
    return ordered[columns].reset_index(drop=True)


def build_plot(curves: dict[str, pd.DataFrame]) -> None:
    fig, (ax_all, ax_phase3) = plt.subplots(
        1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [2.0, 1.0]}
    )

    # --- Left: every tried variation, including the diverged ones ---
    for run in RUNS:
        df = curves[run["label"]]
        ax_all.plot(
            df["epoch"],
            df["val_macro_f1"],
            label=run["label"],
            color=run["color"],
            linestyle=run["linestyle"],
            linewidth=run["linewidth"],
        )

    for _, lo, hi in PHASES:
        ax_all.axvline(hi + 0.5, color="#cccccc", linestyle=":", linewidth=1.0, zorder=0)
    for name, lo, hi in PHASES:
        ax_all.text(
            (lo + hi) / 2,
            0.03,
            name,
            ha="center",
            va="bottom",
            fontsize=8,
            color="#666666",
        )

    ax_all.set_xlabel("Epoch")
    ax_all.set_ylabel("Validation macro F1")
    ax_all.set_title("All sweep runs")
    ax_all.set_xlim(0.5, 50.5)
    ax_all.set_ylim(0.0, 1.02)
    ax_all.grid(alpha=0.25)
    ax_all.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # --- Right: completed runs only, zoomed on the phase-3 region ---
    for run in RUNS:
        if run["status"] != "completed":
            continue
        df = curves[run["label"]]
        window = df[df["epoch"] >= 25]
        ax_phase3.plot(
            window["epoch"],
            window["val_macro_f1"],
            label=run["label"],
            color=run["color"],
            linewidth=run["linewidth"],
        )
        peak = df.loc[df["val_macro_f1"].idxmax()]
        ax_phase3.plot(
            peak["epoch"],
            peak["val_macro_f1"],
            marker="o",
            markersize=6,
            color=run["color"],
            zorder=5,
        )
        if run["experiment"] == "sense_full_lr3e-4_p3lr0.3":
            ax_phase3.annotate(
                f"best {peak['val_macro_f1']:.4f}",
                xy=(peak["epoch"], peak["val_macro_f1"]),
                xytext=(-10, 8),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold",
                color=run["color"],
                ha="right",
            )

    ax_phase3.axvline(30.5, color="#cccccc", linestyle=":", linewidth=1.0, zorder=0)
    ax_phase3.set_xlabel("Epoch")
    ax_phase3.set_ylabel("Validation macro F1")
    ax_phase3.set_title("Completed runs — phase 3 zoom (epochs 25–50)")
    ax_phase3.set_xlim(24.5, 50.5)
    ax_phase3.grid(alpha=0.25)

    fig.suptitle("P7.3 targeted sweep — validation macro F1 by epoch", fontsize=13, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = OUT_DIR / "comparison_plot.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    curves = {run["label"]: _read_metrics(ROOT / run["run_dir"]) for run in RUNS}

    table = build_rows()
    csv_path = OUT_DIR / "summary_table.csv"
    table.to_csv(csv_path, index=False)
    print(f"wrote {csv_path.relative_to(ROOT)}")

    build_plot(curves)

    display = table[
        [
            "rank",
            "experiment",
            "status",
            "best_epoch",
            "best_val_macro_f1",
            "best_val_accuracy",
            "best_val_min_class_accuracy",
            "phase3_mean_macro_f1",
            "wall_time_hms",
        ]
    ]
    print()
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()
