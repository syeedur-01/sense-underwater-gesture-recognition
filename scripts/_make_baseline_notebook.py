"""
scripts/_make_baseline_notebook.py
Phase 4 — P4.4: Generate notebooks/03_baseline_experiment_analysis.ipynb.

Run this script to regenerate the notebook from scratch:
    python scripts/_make_baseline_notebook.py

Then execute it with:
    jupyter nbconvert --to notebook --execute notebooks/03_baseline_experiment_analysis.ipynb \
        --output notebooks/03_baseline_experiment_analysis.ipynb
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def make_code_cell(source: str | list[str]) -> dict:
    if isinstance(source, list):
        source = "\n".join(source)
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def make_markdown_cell(source: str | list[str]) -> dict:
    if isinstance(source, list):
        source = "\n".join(source)
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source,
    }


def build_notebook() -> dict:
    cells = [
        make_markdown_cell(
            [
                "# 03 — Baseline Experiment Analysis\n",
                "**Phase 4 — Baseline Reproduction**\n",
                "This notebook consumes the training metrics and test evaluation artefacts produced by:\n",
                "- `scripts/train_baseline.py` → `runs/baseline/metrics.csv`\n",
                "- `scripts/evaluate_baseline.py` → `runs/baseline/test_confusion.npy`, "
                "`runs/baseline/test_per_class_f1.csv`\n",
                "\n> Run the two scripts above before executing this notebook.",
            ]
        ),
        # ── Imports ────────────────────────────────────────────────────────────
        make_code_cell(
            [
                "import sys, json",
                "from pathlib import Path",
                "",
                "import numpy as np",
                "import pandas as pd",
                "import matplotlib.pyplot as plt",
                "import matplotlib.ticker as mticker",
                "import seaborn as sns",
                "",
                "ROOT = Path().resolve().parent  # assumes notebook is in notebooks/",
                "sys.path.insert(0, str(ROOT))",
                "",
                "RUNS_DIR = ROOT / 'runs' / 'baseline'",
                "PROCESSED_DIR = ROOT / 'data' / 'processed'",
                "",
                "sns.set_theme(style='darkgrid', palette='muted')",
                "print('ROOT:', ROOT)",
            ]
        ),
        # ── 1. Training / validation curves ───────────────────────────────────
        make_markdown_cell("## 1. Training & Validation Curves"),
        make_code_cell(
            [
                "metrics = pd.read_csv(RUNS_DIR / 'metrics.csv')",
                "print(metrics.head())",
                "",
                "fig, axes = plt.subplots(1, 2, figsize=(14, 5))",
                "",
                "# Loss",
                "axes[0].plot(metrics['epoch'], metrics['train_loss'], label='Train Loss', linewidth=2)",
                "axes[0].plot(metrics['epoch'], metrics['val_loss'],   label='Val Loss',   linewidth=2, linestyle='--')",
                "axes[0].set_title('Loss Curves', fontsize=13, fontweight='bold')",
                "axes[0].set_xlabel('Epoch')",
                "axes[0].set_ylabel('Cross-Entropy Loss')",
                "axes[0].legend()",
                "",
                "# Accuracy",
                "axes[1].plot(metrics['epoch'], metrics['train_acc'] * 100, label='Train Acc', linewidth=2)",
                "axes[1].plot(metrics['epoch'], metrics['val_acc']   * 100, label='Val Acc',   linewidth=2, linestyle='--')",
                "axes[1].yaxis.set_major_formatter(mticker.PercentFormatter())",
                "axes[1].set_title('Accuracy Curves', fontsize=13, fontweight='bold')",
                "axes[1].set_xlabel('Epoch')",
                "axes[1].set_ylabel('Accuracy (%)')",
                "axes[1].legend()",
                "",
                "plt.tight_layout()",
                "plt.savefig(RUNS_DIR / 'training_curves.png', dpi=150, bbox_inches='tight')",
                "plt.show()",
            ]
        ),
        # ── 2. Checkpoint analysis ─────────────────────────────────────────────
        make_markdown_cell("## 2. Checkpoint Analysis"),
        make_code_cell(
            [
                "best_row = metrics.loc[metrics['val_acc'].idxmax()]",
                "print(f'Best epoch       : {int(best_row[\"epoch\"])}')",
                "print(f'Best val accuracy: {best_row[\"val_acc\"]*100:.2f}%')",
                "print(f'Best val loss    : {best_row[\"val_loss\"]:.4f}')",
                "",
                "fig, ax = plt.subplots(figsize=(10, 4))",
                "ax.plot(metrics['epoch'], metrics['val_acc'] * 100, color='steelblue', linewidth=2, label='Val Acc')",
                "ax.axvline(best_row['epoch'], color='crimson', linestyle='--', label=f'Best epoch={int(best_row[\"epoch\"])}')",
                "ax.axhline(best_row['val_acc'] * 100, color='orange', linestyle=':', alpha=0.8)",
                "ax.set_xlabel('Epoch')",
                "ax.set_ylabel('Val Accuracy (%)')",
                "ax.set_title('Validation Accuracy vs Epoch — Best Checkpoint Highlighted', fontweight='bold')",
                "ax.yaxis.set_major_formatter(mticker.PercentFormatter())",
                "ax.legend()",
                "plt.tight_layout()",
                "plt.savefig(RUNS_DIR / 'val_acc_checkpoint.png', dpi=150, bbox_inches='tight')",
                "plt.show()",
            ]
        ),
        # ── 3. Test performance summary table ─────────────────────────────────
        make_markdown_cell("## 3. Baseline Performance Summary (vs Yang et al.)"),
        make_code_cell(
            [
                "per_class_f1 = pd.read_csv(RUNS_DIR / 'test_per_class_f1.csv')",
                "",
                "overall_acc  = per_class_f1  # placeholder; read from results md",
                "",
                "# Parse accuracy + macro F1 from baseline_results.md",
                "results_md = (PROCESSED_DIR / 'baseline_results.md').read_text(encoding='utf-8')",
                "import re",
                "acc_match  = re.search(r'Test Accuracy.*?(\\d+\\.\\d+)%', results_md)",
                "f1_match   = re.search(r'Macro F1-score.*?(\\d+\\.\\d+)', results_md)",
                "our_acc  = float(acc_match.group(1))  if acc_match  else None",
                "our_f1   = float(f1_match.group(1))   if f1_match   else None",
                "",
                "summary = pd.DataFrame({",
                "    'Model'         : ['Yang et al. - GoogleLeNet', 'Yang et al. - ResNet', 'Our ResNet18 Baseline'],",
                "    'Accuracy (%)'  : [90.08, 88.59, our_acc],",
                "    'Macro F1'      : ['N/A', 'N/A', f'{our_f1:.4f}' if our_f1 else 'N/A'],",
                "    'Notes'         : ['Literature result', 'Literature result', 'Standard split, left-image only'],",
                "})",
                "print(summary.to_string(index=False))",
                "display(summary)",
            ]
        ),
        # ── 4. Confusion matrix ────────────────────────────────────────────────
        make_markdown_cell("## 4. Test Confusion Matrix"),
        make_code_cell(
            [
                "cm = np.load(RUNS_DIR / 'test_confusion.npy')",
                "",
                "class_names = [",
                "    'start_comm', 'end_comm', 'up', 'down', 'photo', 'backwards',",
                "    'carry', 'boat', 'here', 'mosaic', 'num_delimiter', 'one',",
                "    'two', 'three', 'four', 'five', 'true_neg',",
                "]",
                "class_names = class_names[:cm.shape[0]]",
                "",
                "fig, ax = plt.subplots(figsize=(14, 12))",
                "cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True).clip(min=1)",
                "sns.heatmap(",
                "    cm_norm, annot=True, fmt='.2f', cmap='Blues',",
                "    xticklabels=class_names, yticklabels=class_names,",
                "    ax=ax, linewidths=0.3,",
                ")",
                "ax.set_title('Normalised Confusion Matrix — ResNet18 Baseline (Test Set)', fontsize=14, fontweight='bold')",
                "ax.set_xlabel('Predicted', fontsize=12)",
                "ax.set_ylabel('True', fontsize=12)",
                "plt.xticks(rotation=45, ha='right', fontsize=9)",
                "plt.yticks(rotation=0, fontsize=9)",
                "plt.tight_layout()",
                "plt.savefig(RUNS_DIR / 'confusion_matrix.png', dpi=150, bbox_inches='tight')",
                "plt.show()",
            ]
        ),
        # ── 5. Per-class F1 ────────────────────────────────────────────────────
        make_markdown_cell("## 5. Per-Class F1 Scores"),
        make_code_cell(
            [
                "fig, ax = plt.subplots(figsize=(12, 5))",
                "colors = sns.color_palette('muted', n_colors=len(per_class_f1))",
                "bars = ax.bar(per_class_f1['class_name'], per_class_f1['f1'], color=colors)",
                "ax.set_title('Per-Class F1 Score - ResNet18 Baseline (Test Set)', fontweight='bold')",
                "ax.set_xlabel('Gesture Class')",
                "ax.set_ylabel('F1 Score')",
                "ax.set_ylim(0, 1.0)",
                "plt.xticks(rotation=45, ha='right', fontsize=9)",
                "plt.tight_layout()",
                "plt.savefig(RUNS_DIR / 'per_class_f1.png', dpi=150, bbox_inches='tight')",
                "plt.show()",
            ]
        ),
        make_markdown_cell(
            [
                "---",
                "*End of Phase 4 Baseline Analysis.*",
                "",
                "**Next step:** Proceed to Phase 5 — SENSE Model Construction,",
                "provided the test accuracy above is non-trivial (>20%).",
            ]
        ),
    ]

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
            },
        },
        "cells": cells,
    }


def main() -> None:
    out_path = ROOT / "notebooks" / "03_baseline_experiment_analysis.ipynb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nb = build_notebook()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Notebook written -> {out_path}")
    print("Execute with:")
    print(
        "  jupyter nbconvert --to notebook --execute --inplace "
        "notebooks/03_baseline_experiment_analysis.ipynb"
    )


if __name__ == "__main__":
    main()
