"""Cross-scenario (LOSO) aggregation for the robustness study.

This module turns the per-fold artefacts produced by ``scripts/evaluate.py`` into
one per-scenario table and one statistical summary. It performs no inference and
reads no images; every number it reports comes from a fold's ``metrics.json`` and
``per_class.csv``.

Per-fold macro F1 is computed over the classes that the fold's test partition
actually contains (decision D21). Five of the eight held-out scenarios do not
record every gesture, so a seventeen-class macro F1 is capped by class coverage
rather than by model quality: ``brodarski-A`` contains eight of the seventeen
classes, which caps that fold at 0.4706 however well it classifies. Reporting the
present-class mean keeps folds comparable. The seventeen-class value and the
coverage are reported beside it so that the difference is visible rather than
hidden.

The aggregate is reported as a mean, a sample standard deviation, and a 95%
confidence interval for the mean computed from the t distribution, which is
appropriate for eight folds.

Typical use, from ``scripts/run_cross_scenario.py``::

    records = [load_fold(fold, eval_dir, scenario) for fold, eval_dir, scenario in folds]
    table = build_table(records)
    write_outputs(table, Path("results/experiment_2_cross_scenario"))
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "FOLD_COUNT",
    "class_coverage",
    "load_fold",
    "build_table",
    "summarise",
    "write_outputs",
    "NUM_CLASSES",
]

#: Number of LOSO folds defined by the split protocol.
FOLD_COUNT = 8

#: Total number of classes in the task, including the no-gesture class.
NUM_CLASSES = 17

#: Metrics summarised across folds. Accuracy is the primary cross-fold quantity
#: because it does not depend on how many classes a scenario contains.
SUMMARY_METRICS = (
    ("accuracy", "Accuracy"),
    ("macro_f1_present", "Macro F1 (present classes)"),
    ("macro_f1_17class", "Macro F1 (all 17 classes)"),
    ("min_class_accuracy", "Minimum class accuracy"),
)


def class_coverage(per_class: pd.DataFrame) -> dict[str, Any]:
    """Describe which classes a fold's test partition actually contains.

    Args:
        per_class: Per-class table with ``class_id``, ``class_name`` and
            ``support`` columns, as written by ``scripts/evaluate.py``.

    Returns:
        Dictionary with the number of present classes, the number of absent
        classes, the fraction present, and the absent class names.
    """
    present = per_class.loc[per_class["support"] > 0]
    absent = per_class.loc[per_class["support"] == 0]
    return {
        "classes_present": int(len(present)),
        "classes_absent": int(len(absent)),
        "coverage": float(len(present) / NUM_CLASSES),
        "absent_classes": list(absent["class_name"]),
    }


def present_class_macro_f1(per_class: pd.DataFrame) -> float:
    """Macro F1 averaged over the classes the partition contains.

    A class with zero support has no F1 to average, so it is excluded rather than
    scored as zero. On a partition containing every class this equals the
    seventeen-class macro F1.

    Args:
        per_class: Per-class table with ``support`` and ``f1`` columns.

    Returns:
        The mean F1 over classes with positive support, or NaN if there are none.
    """
    present = per_class.loc[per_class["support"] > 0, "f1"]
    if present.empty:
        return float("nan")
    return float(present.mean())


def load_fold(fold: int, eval_dir: Path | str, scenario: str) -> dict[str, Any]:
    """Load one fold's evaluation artefacts into a record.

    Args:
        fold: One-based fold index.
        eval_dir: Directory holding ``metrics.json`` and ``per_class.csv``.
        scenario: Name of the held-out scenario for this fold.

    Returns:
        A flat record suitable for :func:`build_table`.

    Raises:
        FileNotFoundError: If either artefact is missing.
    """
    directory = Path(eval_dir)
    metrics_path = directory / "metrics.json"
    per_class_path = directory / "per_class.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing metrics.json for fold {fold}: {metrics_path}")
    if not per_class_path.exists():
        raise FileNotFoundError(f"missing per_class.csv for fold {fold}: {per_class_path}")

    bundle = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics = bundle["metrics"]
    per_class = pd.read_csv(per_class_path)

    coverage = class_coverage(per_class)
    n_samples = int(per_class["support"].sum())

    return {
        "fold": int(fold),
        "scenario": scenario,
        "n_test_pairs": n_samples,
        "classes_present": coverage["classes_present"],
        "classes_absent": coverage["classes_absent"],
        "coverage": coverage["coverage"],
        "absent_classes": ", ".join(coverage["absent_classes"]),
        "accuracy": float(metrics["overall_accuracy"]),
        "macro_f1_present": present_class_macro_f1(per_class),
        "macro_f1_17class": float(metrics["macro_f1"]),
        "min_class_accuracy": float(metrics["min_class_accuracy"]),
        "correct": int(round(metrics["overall_accuracy"] * n_samples)),
        "checkpoint_sha256": bundle.get("provenance", {}).get("checkpoint_sha256", ""),
        "split_sha256": bundle.get("provenance", {}).get("split_sha256", ""),
        "eval_dir": str(directory),
    }


def summarise(values: Iterable[float]) -> dict[str, float]:
    """Mean, sample standard deviation, and 95% t-interval for the mean.

    Args:
        values: Per-fold values of one metric.

    Returns:
        Dictionary with ``n``, ``mean``, ``sd``, ``ci_low`` and ``ci_high``. With
        fewer than two folds the spread and interval are NaN, because a single
        fold carries no information about variability.
    """
    arr = np.asarray([v for v in values if not math.isnan(v)], dtype=float)
    n = int(arr.size)
    if n == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "sd": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }

    mean = float(arr.mean())
    if n < 2:
        return {
            "n": n,
            "mean": mean,
            "sd": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }

    sd = float(arr.std(ddof=1))
    half_width = float(stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n))
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "ci_low": mean - half_width,
        "ci_high": mean + half_width,
    }


def build_table(records: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Assemble fold records into the per-scenario results table.

    Rows are ordered by fold index. The table keeps the seventeen-class macro F1
    and the coverage next to the present-class value so that the effect of class
    coverage stays visible in every downstream report.
    """
    columns = [
        "fold",
        "scenario",
        "n_test_pairs",
        "classes_present",
        "classes_absent",
        "coverage",
        "absent_classes",
        "accuracy",
        "macro_f1_present",
        "macro_f1_17class",
        "min_class_accuracy",
        "correct",
        "checkpoint_sha256",
        "split_sha256",
        "eval_dir",
    ]
    table = pd.DataFrame(list(records))
    if table.empty:
        return pd.DataFrame(columns=columns)
    return table.sort_values("fold").reset_index(drop=True)[columns]


def _summary_table(table: pd.DataFrame) -> pd.DataFrame:
    """Build the mean, SD, and confidence-interval table across folds."""
    rows = []
    for key, label in SUMMARY_METRICS:
        stats_row = summarise(table[key].tolist())
        rows.append({"metric": label, **stats_row})
    return pd.DataFrame(rows)


def _format(value: float, digits: int = 4) -> str:
    """Format a float for the Markdown summary, or a dash when undefined."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:.{digits}f}"


def write_outputs(
    table: pd.DataFrame,
    out_dir: Path | str,
    *,
    checkpoint_note: str | None = None,
) -> tuple[Path, Path]:
    """Write the per-scenario table and the statistical summary.

    Args:
        table: Output of :func:`build_table`.
        out_dir: Destination directory, created if absent.
        checkpoint_note: Optional sentence identifying the evaluated checkpoints,
            included in the summary so that every fold traces to its own run.

    Returns:
        The paths of ``per_scenario_results.csv`` and ``statistical_summary.md``.
    """
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)

    csv_path = directory / "per_scenario_results.csv"
    table.to_csv(csv_path, index=False, float_format="%.6f")

    md_path = directory / "statistical_summary.md"
    md_path.write_text(_render_summary(table, checkpoint_note=checkpoint_note), encoding="utf-8")
    return csv_path, md_path


def _render_summary(table: pd.DataFrame, *, checkpoint_note: str | None) -> str:
    """Render the Markdown statistical summary."""
    summary = _summary_table(table)
    folds_run = int(len(table))
    expected = FOLD_COUNT
    coverage = table["coverage"]

    lines = [
        "# Cross-scenario (LOSO) statistical summary",
        "",
        f"Folds reported: {folds_run} of {expected}.",
        "",
        "Per-fold macro F1 is averaged over the classes that the fold's test partition",
        "contains (D21). The seventeen-class value is reported beside it, together with",
        "class coverage, because five of the eight held-out scenarios do not record every",
        "gesture and a seventeen-class macro F1 is therefore capped by coverage rather than",
        "by model quality.",
        "",
    ]
    if checkpoint_note:
        lines += [checkpoint_note, ""]
    if folds_run < expected:
        lines += [
            f"**Incomplete.** Only {folds_run} of {expected} folds have been evaluated, so the",
            "aggregate below is not the eight-fold result and must not be reported as one.",
            "",
        ]

    lines += [
        "## Aggregate across folds",
        "",
        "| Metric | Folds | Mean | SD | 95% CI for the mean |",
        "|---|---:|---:|---:|---|",
    ]
    for _, row in summary.iterrows():
        ci = (
            f"[{_format(row['ci_low'])}, {_format(row['ci_high'])}]"
            if not math.isnan(row["ci_low"])
            else "-"
        )
        lines.append(
            f"| {row['metric']} | {int(row['n'])} | {_format(row['mean'])} | "
            f"{_format(row['sd'])} | {ci} |"
        )

    lines += [
        "",
        "The confidence interval is computed from the t distribution with n - 1 degrees of",
        "freedom. With eight folds it is wide; it is reported as a description of spread",
        "across scenarios, not as an estimate of run-to-run variability, which is the",
        "single-seed limitation recorded separately.",
        "",
        "## Per-fold results",
        "",
        "| Fold | Held-out scenario | Test pairs | Classes present | Accuracy | "
        "Macro F1 (present) | Macro F1 (17) | Min class accuracy |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"| {int(row['fold'])} | {row['scenario']} | {int(row['n_test_pairs']):,} | "
            f"{int(row['classes_present'])}/17 | {_format(row['accuracy'])} | "
            f"{_format(row['macro_f1_present'])} | {_format(row['macro_f1_17class'])} | "
            f"{_format(row['min_class_accuracy'])} |"
        )

    lines += [
        "",
        "## Class coverage",
        "",
        f"Mean coverage across folds: {_format(float(coverage.mean()))} of the seventeen",
        "classes. Folds with incomplete coverage are scored on what they contain; the",
        "absent classes are listed per fold in `per_scenario_results.csv`.",
        "",
    ]

    smallest = table.loc[table["n_test_pairs"].idxmin()]
    largest = table.loc[table["n_test_pairs"].idxmax()]
    lines += [
        f"The smallest fold is `{smallest['scenario']}` with "
        f"{int(smallest['n_test_pairs']):,} test pairs, and the largest is "
        f"`{largest['scenario']}` with {int(largest['n_test_pairs']):,}. The smallest fold",
        "contributes most of the variance across folds and is reported with its sample",
        "size attached.",
        "",
    ]
    return "\n".join(lines)
