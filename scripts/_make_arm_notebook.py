"""
scripts/_make_arm_notebook.py
Generate the per-arm experiment-analysis notebooks.

One code path produces one notebook per arm, so the two cannot drift apart:

    03_baseline_experiment_analysis.ipynb   single-image ResNet-18 baseline
    04_control_experiment_analysis.ipynb    backbone-matched control
                                            (EfficientNet-B4, one branch, no text
                                            guidance, no enhancement)

Each notebook reads only artefacts that the training loop and the shared
evaluation entry point produced. Nothing here trains, selects, or tunes.

Regenerate:
    python scripts/_make_arm_notebook.py all
    python scripts/_make_arm_notebook.py baseline

Execute:
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/03_baseline_experiment_analysis.ipynb
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ARMS: dict[str, dict[str, str]] = {
    "baseline": {
        "notebook": "03_baseline_experiment_analysis",
        "run_parent": "runs/baseline_resnet18_trainer",
        "eval_dir": "results/experiment_1_standard/baseline",
        "heading": "03 - Baseline Experiment Analysis",
        "short": "baseline",
        "display": "Single-image ResNet-18 baseline",
        "description": (
            "The baseline is the only baseline used in any comparison. It was trained by the\n"
            "same training loop, with the same checkpoint-selection rule and the same metric\n"
            "implementation as the proposed model, and it was evaluated on the standard test\n"
            "partition by the same evaluation code.\n"
        ),
        "note": (
            "'This notebook measures the baseline. Whether it differs from the proposed model by more',\n"
            "    'than single-seed variability is decided in the comparison notebook and in the results',\n"
            "    'record, not here.',\n"
        ),
    },
    "control": {
        "notebook": "04_control_experiment_analysis",
        "run_parent": "runs/control_selected_recipe",
        "eval_dir": "results/experiment_1_standard/control",
        "heading": "04 - Control Experiment Analysis",
        "short": "control",
        "display": "Backbone-matched control (EfficientNet-B4, one branch, no text guidance, no enhancement)",
        "description": (
            "The control uses the same backbone, schedule, loss, and seed as the selected\n"
            "configuration, with stereo, text guidance, and enhancement disabled. It therefore\n"
            "isolates the contribution of those three modules at the recipe that was actually\n"
            "chosen, rather than at the reference recipe used by the earlier control.\n"
        ),
        "note": (
            "'This notebook measures the control. The size of the control-to-proposed-model gap is',\n"
            "    'read in the comparison notebook and against the pre-specified interpretation rule,',\n"
            "    'not here.',\n"
        ),
    },
}


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
        # Elements carry their own trailing newline, so joining with a newline
        # would double every blank line. Strip one trailing newline per element
        # and let the join supply the separator.
        source = "\n".join(part[:-1] if part.endswith("\n") else part for part in source)
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source,
    }


def build_notebook(arm_key: str) -> dict:
    arm = ARMS[arm_key]
    run_parent = arm["run_parent"]
    eval_dir = arm["eval_dir"]
    title = arm["heading"]
    short = arm["short"]
    display = arm["display"]

    cells = [
        make_markdown_cell(
            [
                f"# {title}\n",
                "\n",
                f"**{display}, shared protocol**\n",
                "\n",
                arm["description"],
                "\n",
                "| Input | Partition | Produced by |\n",
                "|---|---|---|\n",
                "| `metrics.csv` | validation | training loop |\n",
                "| `metrics.json`, `per_class.csv`, `confusion_matrix.npy` | standard test | shared evaluation entry point |\n",
                "\n",
                "Nothing in this notebook selects a model or tunes a threshold. The checkpoint was\n",
                "selected during training, before the test partition was read, and it is reproduced\n",
                "here only to make the selection rule visible.\n",
                "\n",
                "> Run the training and evaluation steps for this arm before executing the notebook.",
            ]
        ),
        # ── Setup ─────────────────────────────────────────────────────────────
        make_code_cell(
            [
                "import json",
                "import sys",
                "from pathlib import Path",
                "",
                "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()",
                "sys.path.insert(0, str(ROOT))",
                "",
                "# The project's metrics module selects a non-interactive backend when it is",
                "# imported, so that headless scripts can use it. Import it first, then re-enable",
                "# the inline backend so that figures render in the notebook.",
                "from src.evaluation.metrics import is_better_checkpoint",
                "",
                "%matplotlib inline",
                "",
                "import matplotlib.pyplot as plt",
                "import matplotlib.ticker as mticker",
                "import numpy as np",
                "import pandas as pd",
                "import seaborn as sns",
                "import yaml",
                "from sklearn.metrics import classification_report",
                "",
                f"RUN_PARENT = ROOT / '{run_parent}'",
                "RUN_DIR = sorted(p for p in RUN_PARENT.iterdir() if p.is_dir())[-1]",
                f"EVAL_DIR = ROOT / '{eval_dir}'",
                "",
                "sns.set_theme(style='whitegrid', context='notebook')",
                "pd.set_option('display.width', 200)",
                "",
                "print('run directory :', RUN_DIR.relative_to(ROOT))",
                "print('eval directory:', EVAL_DIR.relative_to(ROOT))",
            ]
        ),
        # ── 1. Training and validation curves ────────────────────────────────
        make_markdown_cell(
            [
                "## 1. Training and validation curves\n",
                "\n",
                "Loss, accuracy, macro F1, and minimum class accuracy against epoch, over the arm's\n",
                "full budget. The shaded background marks the recorded training stages: the backbone\n",
                "is frozen in the first stage and unfrozen in the second.\n",
                "\n",
                "Training accuracy is not recorded by the training loop, so the accuracy panel shows\n",
                "the validation curve only. Accuracy is plotted for completeness. It is not the\n",
                "selection criterion, and the last two panels are what the selection rule reads.",
            ]
        ),
        make_code_cell(
            [
                "metrics = pd.read_csv(RUN_DIR / 'metrics.csv').sort_values('epoch').reset_index(drop=True)",
                "cfg = yaml.safe_load((RUN_DIR / 'config.yaml').read_text(encoding='utf-8'))",
                "",
                "stages = sorted(",
                "    cfg['training']['phases'].items(),",
                "    key=lambda item: item[1]['epochs'][0],",
                ")",
                "stage_spans = [",
                "    (info['epochs'][0], min(info['epochs'][1], int(metrics['epoch'].max())))",
                "    for _, info in stages",
                "]",
                "stage_spans = [span for span in stage_spans if span[0] <= span[1]]",
                "",
                "fig, axes = plt.subplots(2, 2, figsize=(14, 9))",
                "panels = [",
                "    ('train_loss', 'val_loss', 'Loss', 'Cross-entropy loss', None),",
                "    (None, 'val_accuracy', 'Accuracy', 'Accuracy', 'percent'),",
                "    (None, 'val_macro_f1', 'Macro F1', 'Macro F1', None),",
                "    (None, 'val_min_class_accuracy', 'Minimum class accuracy', 'Minimum class accuracy', None),",
                "]",
                "",
                "for ax, (train_col, val_col, title_text, ylabel, fmt) in zip(axes.ravel(), panels):",
                "    for k, (start, end) in enumerate(stage_spans):",
                "        if k % 2 == 0:",
                "            ax.axvspan(start - 0.5, end + 0.5, color='0.92', zorder=0)",
                "    if train_col is not None:",
                "        ax.plot(metrics['epoch'], metrics[train_col], label='train', linewidth=2)",
                "    if val_col in metrics.columns:",
                "        ax.plot(metrics['epoch'], metrics[val_col], label='validation',",
                "                linewidth=2, linestyle='--')",
                "    ax.set_title(title_text, fontsize=12, fontweight='bold')",
                "    ax.set_xlabel('Epoch')",
                "    ax.set_ylabel(ylabel)",
                "    if fmt == 'percent':",
                "        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))",
                "    if ax.get_legend_handles_labels()[0]:",
                "        ax.legend(loc='lower right')",
                "",
                "for k, (start, end) in enumerate(stage_spans):",
                "    axes[0, 0].text(",
                "        (start + end) / 2, axes[0, 0].get_ylim()[1],",
                "        f'stage {k + 1}', ha='center', va='top', fontsize=9, color='0.35',",
                "    )",
                "",
                "fig.suptitle(",
                f"    '{display} - training and validation curves (validation partition)',",
                "    fontsize=13, fontweight='bold',",
                ")",
                "plt.tight_layout()",
                "plt.savefig(EVAL_DIR / 'training_curves.png', dpi=150, bbox_inches='tight')",
                "plt.show()",
                "",
                "metrics[[",
                "    'epoch', 'train_loss', 'val_loss', 'val_accuracy',",
                "    'val_macro_f1', 'val_min_class_accuracy',",
                "]].tail(6).round(4)",
            ]
        ),
        # ── 2. Checkpoint analysis ───────────────────────────────────────────
        make_markdown_cell(
            [
                "## 2. Checkpoint analysis\n",
                "\n",
                "The checkpoint was selected by validation macro F1, with a 0.005 tie window that is\n",
                "broken by minimum class accuracy. The rule is applied epoch by epoch against the\n",
                "best value seen so far, so a later epoch can displace an earlier one even when its\n",
                "macro F1 is slightly lower.\n",
                "\n",
                "The simulation below calls the trainer's own implementation of that comparison, so\n",
                "the epoch it returns is the epoch the trainer saved.\n",
                "\n",
                "The interesting case is a run where the highest macro F1 and the selected checkpoint\n",
                "are different epochs. The plot shows why the rule chose what it chose.",
            ]
        ),
        make_code_cell(
            [
                "best_f1 = best_min_acc = None",
                "selected_epoch = None",
                "",
                "for _, row in metrics.iterrows():",
                "    if is_better_checkpoint(",
                "        row['val_macro_f1'], row['val_min_class_accuracy'], best_f1, best_min_acc",
                "    ):",
                "        best_f1 = float(row['val_macro_f1'])",
                "        best_min_acc = float(row['val_min_class_accuracy'])",
                "        selected_epoch = int(row['epoch'])",
                "",
                "selected = metrics.loc[metrics['epoch'] == selected_epoch].iloc[0]",
                "peak = metrics.loc[metrics['val_macro_f1'].idxmax()]",
                "",
                "print(f'selected epoch      : {selected_epoch}')",
                "print(f'macro F1            : {selected[\"val_macro_f1\"]:.6f}')",
                "print(f'minimum class acc.  : {selected[\"val_min_class_accuracy\"]:.6f}')",
                'print(f\'peak macro F1 epoch : {int(peak["epoch"])} ({peak["val_macro_f1"]:.6f})\')',
                "",
                "window = metrics[",
                "    metrics['val_macro_f1'] >= peak['val_macro_f1'] - 0.005",
                "][['epoch', 'val_macro_f1', 'val_min_class_accuracy']]",
                "print('\\nepochs inside the 0.005 window of the peak macro F1:')",
                "window.round(6)",
            ]
        ),
        make_code_cell(
            [
                "fig, ax = plt.subplots(figsize=(11, 5))",
                "",
                "ax.plot(metrics['epoch'], metrics['val_macro_f1'], marker='o', markersize=3,",
                "        linewidth=2, label='validation macro F1')",
                "ax.plot(metrics['epoch'], metrics['val_min_class_accuracy'], marker='s', markersize=3,",
                "        linewidth=2, linestyle='--', label='minimum class accuracy')",
                "",
                "ax.axhline(peak['val_macro_f1'] - 0.005, color='0.5', linestyle=':', linewidth=1.2,",
                "           label='0.005 window below peak')",
                "ax.axvline(selected_epoch, color='crimson', linewidth=1.5, alpha=0.8,",
                "           label=f'selected epoch {selected_epoch}')",
                "",
                "if int(peak['epoch']) != selected_epoch:",
                "    ax.axvline(int(peak['epoch']), color='darkorange', linestyle='--', linewidth=1.2,",
                "               label=f'peak macro F1 epoch {int(peak[\"epoch\"])}')",
                "",
                "ax.set_xlabel('Epoch')",
                "ax.set_ylabel('Proportion')",
                f"ax.set_title('Checkpoint selection under the macro-F1 rule - {short}', fontweight='bold')",
                "ax.legend(loc='lower right', fontsize=9)",
                "plt.tight_layout()",
                "plt.savefig(EVAL_DIR / 'checkpoint_selection.png', dpi=150, bbox_inches='tight')",
                "plt.show()",
            ]
        ),
        # ── 3. Validation summary ────────────────────────────────────────────
        make_markdown_cell(
            [
                "## 3. Validation performance summary\n",
                "\n",
                "The selected checkpoint, read from the validation record. These values informed\n",
                "model selection and are therefore optimistically biased; the unbiased reading is\n",
                "Section 4 onward.",
            ]
        ),
        make_code_cell(
            [
                "runtime = (RUN_DIR / 'runtime.txt').read_text(encoding='utf-8').strip()",
                "seed = (RUN_DIR / 'seed.txt').read_text(encoding='utf-8').strip()",
                "",
                "summary = pd.DataFrame(",
                "    {",
                "        'Quantity': [",
                "            'Selected epoch',",
                "            'Epochs completed',",
                "            'Macro F1 at selection',",
                "            'Accuracy at selection',",
                "            'Minimum class accuracy at selection',",
                "            'Validation loss at selection',",
                "            'Peak macro F1',",
                "            'Minimum class accuracy at peak epoch',",
                "            'Seed',",
                "            'Training wall time',",
                "        ],",
                "        'Value': [",
                "            str(selected_epoch),",
                '            f\'{int(metrics["epoch"].max())} / {int(cfg["training"]["epochs"])}\',',
                "            f'{selected[\"val_macro_f1\"]:.6f}',",
                "            f'{selected[\"val_accuracy\"]:.6f}',",
                "            f'{selected[\"val_min_class_accuracy\"]:.6f}',",
                "            f'{selected[\"val_loss\"]:.6f}',",
                '            f\'{peak["val_macro_f1"]:.6f} (epoch {int(peak["epoch"])})\',',
                "            f'{peak[\"val_min_class_accuracy\"]:.6f}',",
                "            seed,",
                "            runtime,",
                "        ],",
                "    }",
                ")",
                "summary",
            ]
        ),
        # ── 4. Test partition: composition and confusion ─────────────────────
        make_markdown_cell(
            [
                "## 4. Test partition composition and confusion matrix\n",
                "\n",
                "Standard test partition, 2,465 pairs, one inference pass. The distribution is shown\n",
                "first because every row of the matrix that follows is normalised within its own\n",
                "class: a row with seven samples moves by 14 percentage points when a single image\n",
                "changes side.\n",
                "\n",
                "The matrix is row-normalised, so each row sums to one and the diagonal is the\n",
                "per-class accuracy. The saved artefact is `confusion_matrix_norm.png` in the\n",
                "evaluation directory; the cell redraws it from the stored counts.",
            ]
        ),
        make_code_cell(
            [
                "bundle = json.loads((EVAL_DIR / 'metrics.json').read_text(encoding='utf-8'))",
                "test_metrics = bundle['metrics']",
                "provenance = bundle['provenance']",
                "CLASS_NAMES = bundle['class_names']  # canonical order, as written by the evaluator",
                "cm = np.load(EVAL_DIR / 'confusion_matrix.npy')",
                "per_class = pd.read_csv(EVAL_DIR / 'per_class.csv')",
                "",
                "assert cm.shape == (17, 17), cm.shape",
                "assert int(cm.sum()) == provenance['n_samples'], 'matrix total does not match provenance'",
                "assert int(per_class['support'].sum()) == provenance['n_samples'], 'supports do not sum to n'",
                "assert abs(np.trace(cm) / cm.sum() - test_metrics['overall_accuracy']) < 1e-12",
                "",
                "order = per_class.sort_values('class_id').reset_index(drop=True)",
                "",
                "fig, ax = plt.subplots(figsize=(9, 6.5))",
                "bars = ax.barh(order['class_name'], order['support'], color=sns.color_palette('muted', 17))",
                "for bar, n in zip(bars, order['support']):",
                "    ax.text(bar.get_width() + 4, bar.get_y() + bar.get_height() / 2, f'{int(n):,}',",
                "            va='center', fontsize=8, color='0.25')",
                "ax.invert_yaxis()",
                "ax.set_xlabel('Test samples')",
                "ax.set_xlim(0, order['support'].max() * 1.14)",
                f"ax.set_title('Test-partition class distribution - {short}', fontweight='bold')",
                "plt.tight_layout()",
                "plt.savefig(EVAL_DIR / 'test_class_distribution.png', dpi=150, bbox_inches='tight')",
                "plt.show()",
                "",
                "print(f'classes: {len(order)}  samples: {int(order[\"support\"].sum()):,}')",
                'print(f\'largest class  : {order.loc[order["support"].idxmax(), "class_name"]} \'',
                "      f'(n={int(order[\"support\"].max())})')",
                'print(f\'smallest class : {order.loc[order["support"].idxmin(), "class_name"]} \'',
                "      f'(n={int(order[\"support\"].min())})')",
                "print(f'classes with fewer than 20 samples: {int((order[\"support\"] < 20).sum())}')",
            ]
        ),
        make_code_cell(
            [
                "cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)",
                "",
                "fig, ax = plt.subplots(figsize=(13, 11))",
                "sns.heatmap(",
                "    cm_norm, annot=True, fmt='.2f', cmap='Blues',",
                "    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,",
                "    ax=ax, linewidths=0.3, vmin=0.0, vmax=1.0,",
                "    cbar_kws={'label': 'row proportion'},",
                ")",
                "ax.set_title(",
                f"    'Row-normalised confusion matrix - {short} (standard test partition)',",
                "    fontsize=13, fontweight='bold',",
                ")",
                "ax.set_xlabel('Predicted class')",
                "ax.set_ylabel('True class')",
                "plt.xticks(rotation=45, ha='right', fontsize=9)",
                "plt.yticks(rotation=0, fontsize=9)",
                "plt.tight_layout()",
                "plt.show()",
                "",
                "print(f'test accuracy   : {test_metrics[\"overall_accuracy\"]:.6f}')",
                "print(f'test macro F1   : {test_metrics[\"macro_f1\"]:.6f}')",
                "print(f'test min-class  : {test_metrics[\"min_class_accuracy\"]:.6f}')",
                "print(f'correct         : {int(np.trace(cm)):,} / {int(cm.sum()):,}')",
            ]
        ),
        # ── 5. Per-class metrics ─────────────────────────────────────────────
        make_markdown_cell(
            [
                "## 5. Per-class metrics\n",
                "\n",
                "Precision, recall, and F1 per class, with support. The predictions are reconstructed\n",
                "from the stored confusion matrix, which is lossless for these quantities because the\n",
                "matrix counts every (true, predicted) pair.\n",
                "\n",
                "Every class is reported, including the classes with very small support. A high F1 on\n",
                "a class with seven samples is not the same quantity as a high F1 on a class with a\n",
                "thousand, and the support column is what makes the difference visible.",
            ]
        ),
        make_code_cell(
            [
                "y_true, y_pred = [], []",
                "for i in range(cm.shape[0]):",
                "    for j in range(cm.shape[1]):",
                "        count = int(cm[i, j])",
                "        if count:",
                "            y_true.extend([i] * count)",
                "            y_pred.extend([j] * count)",
                "",
                "print(",
                "    classification_report(",
                "        y_true, y_pred,",
                "        labels=list(range(len(CLASS_NAMES))),",
                "        target_names=CLASS_NAMES,",
                "        digits=4,",
                "        zero_division=0,",
                "    )",
                ")",
            ]
        ),
        make_code_cell(
            [
                "per_class = per_class[['class_id', 'class_name', 'support', 'correct', 'accuracy', 'precision', 'recall', 'f1']]",
                "",
                "print(f'support sums to {int(per_class[\"support\"].sum()):,}')",
                "print(f'correct sums to {int(per_class[\"correct\"].sum()):,}')",
                "print(f'classes present : {len(per_class)}')",
                "",
                "per_class.round(4)",
            ]
        ),
        make_markdown_cell(
            [
                "### Precision, recall, and F1 in one view\n",
                "\n",
                "The same numbers as the table above, as a heat map. The colour scale is deliberately\n",
                "truncated rather than spanning zero to one, because every value in this arm sits in a\n",
                "narrow band and a full-range scale would render the whole table a single colour. The\n",
                "printed value is authoritative; the colour is only there to make the pattern visible.",
            ]
        ),
        make_code_cell(
            [
                "metric_cols = ['precision', 'recall', 'f1']",
                "prf = order[metric_cols].to_numpy(dtype=float)",
                "vmin = min(0.5, float(np.floor(prf.min() * 20) / 20))",
                "",
                "fig, ax = plt.subplots(figsize=(6.8, 8))",
                "sns.heatmap(",
                "    prf, annot=True, fmt='.3f', cmap='viridis',",
                "    vmin=vmin, vmax=1.0,",
                "    xticklabels=['Precision', 'Recall', 'F1'],",
                "    yticklabels=order['class_name'],",
                "    ax=ax, linewidths=0.4,",
                "    cbar_kws={'label': f'score (scale {vmin:.2f} - 1.00)'},",
                ")",
                "ax.set_title(",
                f"    'Per-class precision, recall and F1 - {short}',",
                "    fontsize=12, fontweight='bold',",
                ")",
                "ax.set_ylabel('')",
                "plt.tight_layout()",
                "plt.savefig(EVAL_DIR / 'per_class_prf_heatmap.png', dpi=150, bbox_inches='tight')",
                "plt.show()",
                "",
                "print(f'colour scale runs from {vmin:.2f} to 1.00; values below that are pinned to the floor')",
                "print(f'lowest value in the table: {prf.min():.4f}')",
            ]
        ),
        make_markdown_cell(
            [
                "### Most frequent confusions\n",
                "\n",
                "The ten largest off-diagonal counts, with each count shown as a share of the true\n",
                "class. The share matters more than the count: sixteen `end_comm` images read as\n",
                "`start_comm` is eight per cent of that class, while the same count on a class of\n",
                "seven would be the whole class.\n",
                "\n",
                "Pairs whose share is large but whose class is tiny are marked, so that a visually\n",
                "striking bar is not mistaken for the arm's main weakness.",
            ]
        ),
        make_code_cell(
            [
                "off = cm.copy()",
                "np.fill_diagonal(off, 0)",
                "row_support = cm.sum(axis=1)",
                "",
                "flat = np.argsort(off.ravel())[::-1]",
                "pairs = []",
                "for idx in flat:",
                "    i, j = np.unravel_index(idx, cm.shape)",
                "    count = int(cm[i, j])",
                "    if count == 0:",
                "        break",
                "    pairs.append((CLASS_NAMES[i], CLASS_NAMES[j], count, count / row_support[i], int(row_support[i])))",
                "    if len(pairs) == 10:",
                "        break",
                "",
                "pairs = pairs[::-1]  # largest at the top of the chart",
                "labels = [f'{t} -> {p}' for t, p, _, _, _ in pairs]",
                "counts = [c for _, _, c, _, _ in pairs]",
                "shares = [s for _, _, _, s, _ in pairs]",
                "supports = [n for _, _, _, _, n in pairs]",
                "colours = ['#c44e52' if n < 20 else '#4c72b0' for n in supports]",
                "",
                "fig, ax = plt.subplots(figsize=(11, 5.5))",
                "bars = ax.barh(labels, counts, color=colours)",
                "for bar, c, s, n in zip(bars, counts, shares, supports):",
                "    ax.text(bar.get_width() + max(counts) * 0.012, bar.get_y() + bar.get_height() / 2,",
                "            f'{c}  ({s * 100:.1f}% of {n})', va='center', fontsize=8.5)",
                "ax.set_xlabel('Images')",
                "ax.set_xlim(0, max(counts) * 1.35)",
                "ax.set_title(",
                f"    'Ten most frequent confusions - {short} (true class -> predicted class)',",
                "    fontsize=12, fontweight='bold',",
                ")",
                "ax.invert_yaxis()",
                "plt.tight_layout()",
                "plt.savefig(EVAL_DIR / 'top_misclassifications.png', dpi=150, bbox_inches='tight')",
                "plt.show()",
                "",
                "total_errors = int(off.sum())",
                "print(f'errors: {total_errors:,} of {int(cm.sum()):,} ({total_errors / cm.sum() * 100:.2f}%)')",
                "print(f'these ten pairs account for {sum(counts):,} of them ({sum(counts) / total_errors * 100:.1f}%)')",
                "print('red bars: true class has fewer than 20 test samples')",
            ]
        ),
        # ── 6. Per-class chart ───────────────────────────────────────────────
        make_markdown_cell(
            [
                "## 6. Per-class F1 chart\n",
                "\n",
                "F1 by class, with the support of each class printed above its bar. The lowest bars\n",
                "are not necessarily the classes with the most room for improvement; they may simply\n",
                "be the classes with the least evidence.",
            ]
        ),
        make_code_cell(
            [
                "colors = sns.color_palette('muted', n_colors=len(order))",
                "",
                "fig, ax = plt.subplots(figsize=(13, 5.5))",
                "bars = ax.bar(order['class_name'], order['f1'], color=colors)",
                "",
                "for bar, support in zip(bars, order['support']):",
                "    ax.text(",
                "        bar.get_x() + bar.get_width() / 2,",
                "        bar.get_height() + 0.012,",
                "        f'n={int(support)}',",
                "        ha='center', va='bottom', fontsize=8, color='0.3',",
                "    )",
                "",
                "ax.set_ylim(0, 1.08)",
                "ax.set_ylabel('F1 score')",
                "ax.set_xlabel('Class')",
                f"ax.set_title('Per-class F1 - {short} (standard test partition)', fontweight='bold')",
                "ax.axhline(test_metrics['macro_f1'], color='crimson', linestyle='--', linewidth=1.2,",
                "           label=f'macro F1 = {test_metrics[\"macro_f1\"]:.4f}')",
                "plt.xticks(rotation=45, ha='right', fontsize=9)",
                "ax.legend(loc='lower right')",
                "plt.tight_layout()",
                "plt.savefig(EVAL_DIR / 'per_class_f1.png', dpi=150, bbox_inches='tight')",
                "plt.show()",
                "",
                "worst = order.loc[order['f1'].idxmin()]",
                "binding = order.loc[order['accuracy'].idxmin()]",
                'print(f\'lowest F1        : {worst["class_name"]} ({worst["f1"]:.4f}, n={int(worst["support"])})\')',
                'print(f\'lowest accuracy  : {binding["class_name"]} ({binding["accuracy"]:.4f}, n={int(binding["support"])})\')',
            ]
        ),
        # ── 7. Summary ───────────────────────────────────────────────────────
        make_markdown_cell(
            [
                "## 7. Summary\n",
                "\n",
                "What this notebook establishes, and what it does not.",
            ]
        ),
        make_code_cell(
            [
                "lines = [",
                f"    f'{display}: {{provenance[\"parameters\"]:,}} parameters, seed {{seed}}.',",
                "    f'Validation macro F1 at the selected epoch ({selected_epoch}): {selected[\"val_macro_f1\"]:.6f}.',",
                '    f\'Test macro F1: {test_metrics["macro_f1"]:.6f}; test accuracy: {test_metrics["overall_accuracy"]:.6f}.\',',
                '    f\'Test minimum class accuracy: {test_metrics["min_class_accuracy"]:.6f}, attained on "{binding["class_name"]}" with n={int(binding["support"])}.\',',
                '    f\'The lowest per-class F1 is "{worst["class_name"]}" at {worst["f1"]:.4f}, with n={int(worst["support"])}.\',',
                "    f'{total_errors:,} of {int(cm.sum()):,} test images are misclassified, and the ten largest pairs cover {sum(counts) / total_errors * 100:.1f}% of them.',",
                "    '',",
                "    'Reading note: the minimum-class figure is a minimum over seventeen class accuracies, and',",
                "    'the class that attains it may hold very few samples. One prediction moves it',",
                "    'substantially, so it is reported with its support and is not used as a basis for a claim',",
                "    'on its own.',",
                "    '',",
                arm["note"].rstrip(",\n").rstrip() + ",",
                "]",
                "print('\\n'.join(lines))",
            ]
        ),
        make_markdown_cell(
            [
                "---\n",
                "\n",
                "**Artefacts written by this notebook**\n",
                "\n",
                "| File | Content |\n",
                "|---|---|\n",
                "| `training_curves.png` | Loss, accuracy, macro F1, and minimum class accuracy against epoch |\n",
                "| `checkpoint_selection.png` | Selection rule applied to the validation record |\n",
                "| `test_class_distribution.png` | Composition of the test partition by class |\n",
                "| `per_class_prf_heatmap.png` | Precision, recall, and F1 per class |\n",
                "| `top_misclassifications.png` | Ten largest confusion pairs, with share of the true class |\n",
                "| `per_class_f1.png` | Per-class F1 with support annotated |\n",
                "\n",
                "The confusion matrix and the machine-readable metrics are written by the evaluation\n",
                "step, not by this notebook.",
            ]
        ),
    ]

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (sense-venv)",
                "language": "python",
                "name": "sense-venv",
            },
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }


def write_arm(arm_key: str) -> Path:
    arm = ARMS[arm_key]
    out_path = ROOT / "notebooks" / f"{arm['notebook']}.ipynb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nb = build_notebook(arm_key)
    for index, cell in enumerate(nb["cells"]):
        cell.setdefault("id", f"{arm_key[:2]}{index:02d}")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Notebook written -> {out_path.relative_to(ROOT)}")
    return out_path


def main() -> None:
    import sys

    requested = sys.argv[1:] or ["all"]
    keys = list(ARMS) if requested == ["all"] else requested
    for key in keys:
        if key not in ARMS:
            raise SystemExit(f"unknown arm '{key}'; expected one of {list(ARMS)}")
        write_arm(key)
    print("\nExecute with:")
    for key in keys:
        name = ARMS[key]["notebook"]
        print(f"  jupyter nbconvert --to notebook --execute --inplace notebooks/{name}.ipynb")


if __name__ == "__main__":
    main()
