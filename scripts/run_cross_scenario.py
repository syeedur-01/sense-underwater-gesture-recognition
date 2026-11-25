"""Run the eight LOSO fold evaluations and aggregate them.

For each fold this script locates the fold's frozen checkpoint, evaluates it on
that fold's held-out scenario with the shared evaluation entry point, and then
aggregates the eight results into a per-scenario table and a statistical
summary.

The script performs no training and no inference of its own: every number comes
from ``scripts/evaluate.py``, which is the same code that produced the standard
test results. It also never reads a training or validation partition.

Run the folds in any order, or one at a time as they finish training:

    python scripts/run_cross_scenario.py --folds 1 2 3
    python scripts/run_cross_scenario.py --skip-existing

Fold ``i`` is expected at ``runs/loso_fold_i/<timestamp>/checkpoints/best.pth``,
which is where ``scripts/train.py --config configs/loso_fold_i.yaml`` writes it.

Outputs, inside ``--results-dir``:

    fold_1/ ... fold_8/            per-fold artefacts from the evaluation entry point
    per_scenario_results.csv       one row per fold
    statistical_summary.md         mean, SD, and 95% t-interval across folds
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.cross_scenario import (  # noqa: E402
    FOLD_COUNT,
    build_table,
    load_fold,
    write_outputs,
)

#: Directory holding the LOSO split CSVs.
SPLITS_DIR = ROOT / "data" / "processed" / "splits"

#: Default destination for the aggregated results.
DEFAULT_RESULTS_DIR = ROOT / "results" / "experiment_2_cross_scenario"

#: Matches the timestamp directory names written by the training loop.
_TIMESTAMP = re.compile(r"^\d{8}T\d{6}Z$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the eight LOSO fold checkpoints and aggregate the results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=list(range(1, FOLD_COUNT + 1)),
        help="Fold indices to process.",
    )
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory for per-fold artefacts and the aggregate.",
    )
    parser.add_argument(
        "--run-root",
        default=str(ROOT / "runs"),
        help="Directory containing the per-fold training runs.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse a fold's artefacts when they already exist.",
    )
    parser.add_argument("--device", default=None, help="Torch device string, e.g. 'cuda'.")
    parser.add_argument("--batch-size", type=int, default=None, help="Evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Dataloader workers.")
    return parser.parse_args()


def fold_split(fold: int) -> Path:
    """Return the held-out test split for a fold."""
    return SPLITS_DIR / f"scenario_holdout_{fold}_test.csv"


def held_out_scenario(fold: int) -> str:
    """Return the scenario name that fold ``fold`` holds out."""
    split = fold_split(fold)
    if not split.exists():
        raise SystemExit(f"missing LOSO split for fold {fold}: {split}")
    frame = pd.read_csv(split, usecols=["scenario"])
    scenarios = sorted(frame["scenario"].unique())
    if len(scenarios) != 1:
        raise SystemExit(
            f"fold {fold} test split contains {len(scenarios)} scenarios, expected 1: {scenarios}"
        )
    return scenarios[0]


def discover_checkpoint(fold: int, run_root: Path) -> Path | None:
    """Return the newest checkpoint for a fold, or None when the fold has not run."""
    fold_root = run_root / f"loso_fold_{fold}"
    if not fold_root.is_dir():
        return None
    candidates = sorted(
        (d for d in fold_root.iterdir() if d.is_dir() and _TIMESTAMP.match(d.name)),
        key=lambda d: d.name,
    )
    for run_dir in reversed(candidates):
        checkpoint = run_dir / "checkpoints" / "best.pth"
        if checkpoint.exists():
            return checkpoint
    return None


def _display(path: Path) -> str:
    """Render a path relative to the repository when it lies inside it."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def evaluate_fold(fold: int, checkpoint: Path, output_dir: Path, args: argparse.Namespace) -> None:
    """Run the shared evaluation entry point for one fold."""
    scenario = held_out_scenario(fold)
    checkpoint = checkpoint.resolve()
    command = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate.py"),
        "--checkpoint",
        str(checkpoint),
        "--split",
        str(fold_split(fold)),
        "--output",
        str(output_dir.resolve()),
        "--label",
        f"LOSO fold {fold} (held out: {scenario})",
        "--partition-label",
        f"held-out scenario {scenario}",
    ]
    if args.device:
        command += ["--device", args.device]
    if args.batch_size:
        command += ["--batch-size", str(args.batch_size)]
    if args.num_workers is not None:
        command += ["--num-workers", str(args.num_workers)]

    print(f"\n=== fold {fold}: {scenario} ===")
    print(f"  checkpoint: {_display(checkpoint)}")
    completed = subprocess.run(command, cwd=str(ROOT))
    if completed.returncode != 0:
        raise SystemExit(f"evaluation failed for fold {fold} (exit {completed.returncode})")


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir).resolve()
    run_root = Path(args.run_root).resolve()

    evaluated: list[int] = []
    missing: list[int] = []

    for fold in args.folds:
        output_dir = results_dir / f"fold_{fold}"
        if args.skip_existing and (output_dir / "metrics.json").exists():
            print(f"fold {fold}: reusing existing artefacts in {_display(output_dir)}")
            evaluated.append(fold)
            continue

        checkpoint = discover_checkpoint(fold, run_root)
        if checkpoint is None:
            print(
                f"fold {fold}: no checkpoint under {_display(run_root / f'loso_fold_{fold}')} - skipped"
            )
            missing.append(fold)
            continue

        evaluate_fold(fold, checkpoint, output_dir, args)
        evaluated.append(fold)

    if not evaluated:
        raise SystemExit("no fold had both a checkpoint and an evaluation; nothing to aggregate")

    records = [
        load_fold(fold, results_dir / f"fold_{fold}", held_out_scenario(fold)) for fold in evaluated
    ]
    table = build_table(records)
    note = (
        "Each fold's held-out scenario was read once, after that fold's checkpoint was "
        "frozen by the training loop. The same evaluation entry point was used for every "
        "fold, and for the standard test partition."
    )
    csv_path, md_path = write_outputs(table, results_dir, checkpoint_note=note)

    print("\n=== per-fold results ===")
    display = table[
        [
            "fold",
            "scenario",
            "n_test_pairs",
            "classes_present",
            "accuracy",
            "macro_f1_present",
            "macro_f1_17class",
            "min_class_accuracy",
        ]
    ].copy()
    print(display.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    print(f"\nwritten: {_display(csv_path)}")
    print(f"written: {_display(md_path)}")
    if missing:
        print(f"\nincomplete: folds {missing} have no checkpoint yet")
    if len(evaluated) < FOLD_COUNT:
        print(f"note: {len(evaluated)} of {FOLD_COUNT} folds aggregated")


if __name__ == "__main__":
    main()
