"""
scripts/_make_final_evaluation_notebook.py
Generate notebooks/07_final_evaluation.ipynb.

Presents the standard test partition for the three arms of the study:

    selected model, single-image ResNet-18 baseline, backbone-matched control

All three were evaluated by one script, on the same partition, in one inference
pass each, after every configuration and checkpoint decision had been made.
The control is evaluated at the selected recipe, so the comparison between it
and the selected model isolates the three proposed modules.

Regenerate:
    python scripts/_make_final_evaluation_notebook.py

Execute:
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/07_final_evaluation.ipynb
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
        # Elements carry their own trailing newline, so joining with a newline
        # would double every blank line. Strip one trailing newline per element
        # and let the join supply the separator.
        source = "\n".join(part[:-1] if part.endswith("\n") else part for part in source)
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source,
    }


def build_notebook() -> dict:
    cells = [
        make_markdown_cell(
            [
                "# 07 - Final Evaluation on the Standard Test Partition\n",
                "\n",
                "**Three arms, one protocol**\n",
                "\n",
                "This notebook reports the standard test partition for every arm of the study. The\n",
                "partition was read once, after all configuration and checkpoint decisions had been\n",
                "made, by one evaluation script, in a single precision inference pass per model.\n",
                "No checkpoint was re-selected, trained again, or modified.\n",
                "\n",
                "| Arm | Why it is here |\n",
                "|---|---|\n",
                "| Selected model | The proposed configuration |\n",
                "| Single-image ResNet-18 baseline | The reference against which the literature comparison is made |\n",
                "| Backbone-matched control | The same backbone and recipe as the selected model with stereo, text guidance, and enhancement disabled |\n",
                "\n",
                "The control is the arm that carries the attribution argument: because it shares the\n",
                "selected recipe and differs only in the three module switches, a difference between\n",
                "it and the selected model cannot be attributed to the recipe or to backbone capacity.\n",
                "\n",
                "The comparison is read against a noise band fixed before the run: a macro-F1\n",
                "difference smaller than approximately 0.019 is reported as unresolved in either\n",
                "direction, not as a win for whichever model has the higher point estimate.\n",
                "\n",
                "> Run the evaluation step for all three arms before executing this notebook.",
            ]
        ),
        # ── Setup ─────────────────────────────────────────────────────────────
        make_code_cell(
            [
                "%matplotlib inline",
                "",
                "import json",
                "from pathlib import Path",
                "",
                "import matplotlib.pyplot as plt",
                "import numpy as np",
                "import pandas as pd",
                "import seaborn as sns",
                "from matplotlib.colors import TwoSlopeNorm",
                "from sklearn.metrics import classification_report",
                "",
                "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()",
                "",
                "EVAL_ROOT = ROOT / 'results' / 'experiment_1_standard'",
                "ARMATES = {",
                "    'selected': EVAL_ROOT,",
                "    'baseline': EVAL_ROOT / 'baseline',",
                "    'control': EVAL_ROOT / 'control',",
                "}",
                "",
                "sns.set_theme(style='whitegrid', context='notebook')",
                "pd.set_option('display.width', 220)",
                "",
                "# Pre-specified interpretation threshold on macro F1, estimated from the",
                "# run-to-run variability of the training procedure.",
                "NOISE_BAND = 0.019",
                "",
                "# Below this many test samples, a row of a normalised confusion matrix moves by",
                "# more than seven percentage points when one image changes side, so such rows are",
                "# excluded from the difference heat maps.",
                "LOW_SUPPORT = 20",
                "",
                "for name, directory in ARMATES.items():",
                "    status = 'present' if (directory / 'metrics.json').exists() else 'MISSING'",
                "    print(f'{name:<9} {directory.relative_to(ROOT)}  ->  {status}')",
            ]
        ),
        # ── 1. Provenance and integrity ──────────────────────────────────────
        make_markdown_cell(
            [
                "## 1. Provenance and integrity\n",
                "\n",
                "Before any number is read, the artefacts are checked against each other. The\n",
                "confusion matrix must sum to the number of samples evaluated, its diagonal must\n",
                "reproduce the reported accuracy, and the per-class supports must sum to the same\n",
                "total. A metric that fails these checks is not reported.\n",
                "\n",
                "The provenance block records which checkpoint produced each result, its hash, the\n",
                "epoch it came from, the partition, and the numeric precision used.",
            ]
        ),
        make_code_cell(
            [
                "def load_arm(directory):",
                "    bundle = json.loads((directory / 'metrics.json').read_text(encoding='utf-8'))",
                "    return {",
                "        'label': bundle['label'],",
                "        'class_names': bundle['class_names'],",
                "        'metrics': bundle['metrics'],",
                "        'provenance': bundle['provenance'],",
                "        'cm': np.load(directory / 'confusion_matrix.npy'),",
                "        'per_class': pd.read_csv(directory / 'per_class.csv'),",
                "    }",
                "",
                "",
                "arms = {name: load_arm(directory) for name, directory in ARMATES.items()}",
                "reference_names = arms['selected']['class_names']",
                "",
                "for name, arm in arms.items():",
                "    cm = arm['cm']",
                "    m = arm['metrics']",
                "    total = int(cm.sum())",
                "    supports = int(arm['per_class']['support'].sum())",
                "",
                "    assert cm.shape == (17, 17), f'{name}: unexpected matrix shape {cm.shape}'",
                "    assert total == arm['provenance']['n_samples'], f'{name}: matrix total mismatch'",
                "    assert supports == total, f'{name}: per-class supports mismatch'",
                "    assert len(arm['per_class']) == 17, f'{name}: not all classes present'",
                "    assert abs(np.trace(cm) / total - m['overall_accuracy']) < 1e-12, f'{name}: accuracy'",
                "    assert arm['class_names'] == reference_names, f'{name}: class order differs'",
                "",
                "IDENTICAL_SPLIT = len({a['provenance']['split_sha256'] for a in arms.values()}) == 1",
                "print(f'{len(arms)} arms pass integrity checks; identical split hash: {IDENTICAL_SPLIT}')",
                "",
                "provenance_table = pd.DataFrame(",
                "    [",
                "        {",
                "            'arm': arm['label'],",
                "            'epoch': arm['provenance']['checkpoint_epoch'],",
                "            'parameters': f\"{arm['provenance']['parameters']:,}\",",
                "            'samples': arm['provenance']['n_samples'],",
                "            'precision': arm['provenance']['precision'],",
                "            'sha256': arm['provenance']['checkpoint_sha256'][:16] + '...',",
                "        }",
                "        for arm in arms.values()",
                "    ]",
                ")",
                "provenance_table",
            ]
        ),
        # ── 2. Test-partition composition ────────────────────────────────────
        make_markdown_cell(
            [
                "## 2. Composition of the test partition\n",
                "\n",
                "Every row of a normalised confusion matrix is divided by its own class count, so a\n",
                "row with seven samples moves by 14 percentage points when a single image changes\n",
                "side. The distribution is shown once here, because it applies to all three arms.",
            ]
        ),
        make_code_cell(
            [
                "support_order = arms['selected']['per_class'].sort_values('class_id').reset_index(drop=True)",
                "",
                "fig, ax = plt.subplots(figsize=(9, 6.5))",
                "bars = ax.barh(support_order['class_name'], support_order['support'],",
                "               color=sns.color_palette('muted', len(support_order)))",
                "for bar, n in zip(bars, support_order['support']):",
                "    ax.text(bar.get_width() + 4, bar.get_y() + bar.get_height() / 2, f'{int(n):,}',",
                "            va='center', fontsize=8, color='0.25')",
                "ax.invert_yaxis()",
                "ax.set_xlim(0, support_order['support'].max() * 1.14)",
                "ax.set_xlabel('Test samples')",
                "ax.set_title('Standard test partition - class distribution', fontweight='bold')",
                "plt.tight_layout()",
                "plt.savefig(EVAL_ROOT / 'test_class_distribution.png', dpi=150, bbox_inches='tight')",
                "plt.show()",
                "",
                "low_support_classes = support_order[support_order['support'] < LOW_SUPPORT]['class_name'].tolist()",
                'print(f"classes with fewer than {LOW_SUPPORT} samples: {low_support_classes}")',
                "print(f\"smallest support: {int(support_order['support'].min())}\")",
            ]
        ),
        # ── 3. Selected model: confusion matrix ──────────────────────────────
        make_markdown_cell(
            [
                "## 3. Selected model - confusion matrix\n",
                "\n",
                "Rows are normalised, so each row sums to one and the diagonal is the per-class\n",
                "accuracy. Off-diagonal mass shows which classes are confused with which, which is\n",
                "the information an aggregate metric discards.\n",
                "\n",
                "The saved artefact is `confusion_matrix_norm.png` in the evaluation directory; the\n",
                "cell redraws it from the stored counts.",
            ]
        ),
        make_code_cell(
            [
                "sense = arms['selected']",
                "class_names = sense['class_names']",
                "supports = sense['cm'].sum(axis=1)",
                "",
                "",
                "def normalise(cm):",
                "    return cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)",
                "",
                "",
                "cm_norm = normalise(sense['cm'])",
                "",
                "fig, ax = plt.subplots(figsize=(13, 11))",
                "sns.heatmap(",
                "    cm_norm, annot=True, fmt='.2f', cmap='Blues',",
                "    xticklabels=class_names, yticklabels=class_names,",
                "    ax=ax, linewidths=0.3, vmin=0.0, vmax=1.0,",
                "    cbar_kws={'label': 'row proportion'},",
                ")",
                "ax.set_title('Row-normalised confusion matrix - selected model', fontsize=13, fontweight='bold')",
                "ax.set_xlabel('Predicted class')",
                "ax.set_ylabel('True class')",
                "plt.xticks(rotation=45, ha='right', fontsize=9)",
                "plt.yticks(rotation=0, fontsize=9)",
                "plt.tight_layout()",
                "plt.show()",
                "",
                "off = sense['cm'].copy()",
                "np.fill_diagonal(off, 0)",
                "trace = int(np.trace(sense['cm']))",
                'print(f\'test accuracy : {sense["metrics"]["overall_accuracy"]:.6f}\')',
                'print(f\'test macro F1 : {sense["metrics"]["macro_f1"]:.6f}\')',
                "print(f'correct       : {trace:,} / {int(sense[\"cm\"].sum()):,}')",
            ]
        ),
        # ── 4. Selected model: classification report ─────────────────────────
        make_markdown_cell(
            [
                "## 4. Selected model - classification report\n",
                "\n",
                "Precision, recall, and F1 per class with support. Predictions are reconstructed from\n",
                "the stored confusion matrix, which is lossless for these quantities because the\n",
                "matrix counts every (true, predicted) pair.\n",
                "\n",
                "Every class is reported. The rare classes carry the same weight in macro F1 as the\n",
                "frequent ones, so a reader comparing macro F1 across studies needs the support\n",
                "column to know how much evidence stands behind each term.",
            ]
        ),
        make_code_cell(
            [
                "def reconstruct_labels(cm):",
                "    y_true, y_pred = [], []",
                "    for i in range(cm.shape[0]):",
                "        for j in range(cm.shape[1]):",
                "            count = int(cm[i, j])",
                "            if count:",
                "                y_true.extend([i] * count)",
                "                y_pred.extend([j] * count)",
                "    return y_true, y_pred",
                "",
                "",
                "y_true, y_pred = reconstruct_labels(sense['cm'])",
                "print(",
                "    classification_report(",
                "        y_true, y_pred,",
                "        labels=list(range(len(class_names))),",
                "        target_names=class_names,",
                "        digits=4,",
                "        zero_division=0,",
                "    )",
                ")",
            ]
        ),
        make_code_cell(
            [
                "per_class = sense['per_class'].copy()",
                "per_class = per_class[",
                "    ['class_id', 'class_name', 'support', 'correct', 'accuracy', 'precision', 'recall', 'f1']",
                "]",
                "",
                "print(f\"support sums to {int(per_class['support'].sum()):,}\")",
                "print(f\"correct sums to {int(per_class['correct'].sum()):,}\")",
                "per_class.round(4)",
            ]
        ),
        # ── 5. Selected model: per-class charts ──────────────────────────────
        make_markdown_cell(
            [
                "## 5. Selected model - per-class F1 and the three per-class metrics\n",
                "\n",
                "F1 by class, with the support printed above each bar and the macro F1 marked. The\n",
                "gap between a bar and the reference line is not a measure of how much a class could\n",
                "be improved; a short bar on a small class may simply be a short bar on little\n",
                "evidence.\n",
                "\n",
                "The heat map on the right of the second cell shows the same numbers alongside\n",
                "precision and recall, on a truncated colour scale, because every value sits in a\n",
                "narrow band and a full-range scale would render the table a single colour.",
            ]
        ),
        make_code_cell(
            [
                "order = per_class.sort_values('class_id').reset_index(drop=True)",
                "colors = sns.color_palette('muted', n_colors=len(order))",
                "",
                "fig, ax = plt.subplots(figsize=(13, 5.5))",
                "bars = ax.bar(order['class_name'], order['f1'], color=colors)",
                "for bar, support in zip(bars, order['support']):",
                "    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,",
                "            f'n={int(support)}', ha='center', va='bottom', fontsize=8, color='0.3')",
                "ax.set_ylim(0, 1.08)",
                "ax.set_ylabel('F1 score')",
                "ax.set_xlabel('Class')",
                "ax.set_title('Per-class F1 - selected model (standard test partition)', fontweight='bold')",
                "ax.axhline(sense['metrics']['macro_f1'], color='crimson', linestyle='--', linewidth=1.2,",
                "           label=f\"macro F1 = {sense['metrics']['macro_f1']:.4f}\")",
                "plt.xticks(rotation=45, ha='right', fontsize=9)",
                "ax.legend(loc='lower right')",
                "plt.tight_layout()",
                "plt.savefig(EVAL_ROOT / 'per_class_f1.png', dpi=150, bbox_inches='tight')",
                "plt.show()",
                "",
                "worst = order.loc[order['f1'].idxmin()]",
                "binding = order.loc[order['accuracy'].idxmin()]",
                "print(f\"lowest F1       : {worst['class_name']} ({worst['f1']:.4f}, n={int(worst['support'])})\")",
                "print(f\"lowest accuracy : {binding['class_name']} ({binding['accuracy']:.4f}, n={int(binding['support'])})\")",
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
                "    prf, annot=True, fmt='.3f', cmap='viridis', vmin=vmin, vmax=1.0,",
                "    xticklabels=['Precision', 'Recall', 'F1'], yticklabels=order['class_name'],",
                "    ax=ax, linewidths=0.4, cbar_kws={'label': f'score (scale {vmin:.2f} - 1.00)'},",
                ")",
                "ax.set_title('Per-class precision, recall and F1 - selected model', fontsize=12, fontweight='bold')",
                "ax.set_ylabel('')",
                "plt.tight_layout()",
                "plt.savefig(EVAL_ROOT / 'per_class_prf_heatmap.png', dpi=150, bbox_inches='tight')",
                "plt.show()",
                "",
                "print(f'colour scale starts at {vmin:.2f}; lowest value in the table is {prf.min():.4f}')",
            ]
        ),
        # ── 6. Three-arm matrices and differences ────────────────────────────
        make_markdown_cell(
            [
                "## 6. Confusion matrices for all three arms, and the differences between them\n",
                "\n",
                "The three matrices share class order and colour scale, so a row can be compared\n",
                "across arms without re-reading the axes.\n",
                "\n",
                "The difference panels show the selected model minus the other arm, in percentage\n",
                "points of the row. Blue means the selected model recovers images that the other arm\n",
                "loses; red means the reverse. Rows with fewer than twenty test samples are excluded,\n",
                "because in such a row one image is worth more than five percentage points and the\n",
                "colour would reflect the sample size rather than the model.",
            ]
        ),
        make_code_cell(
            [
                "fig, axes = plt.subplots(1, 3, figsize=(21, 7.2))",
                "for ax, (name, arm) in zip(axes, arms.items()):",
                "    sns.heatmap(",
                "        normalise(arm['cm']), annot=True, fmt='.2f', cmap='Blues',",
                "        xticklabels=class_names, yticklabels=class_names,",
                "        vmin=0.0, vmax=1.0, ax=ax, linewidths=0.25,",
                "        annot_kws={'fontsize': 6},",
                "        cbar=(name == 'control'),",
                "        cbar_kws={'label': 'row proportion'},",
                "    )",
                "    ax.set_title(f\"{arm['label']}\\nmacro F1 {arm['metrics']['macro_f1']:.4f}\", fontsize=10)",
                "    ax.set_xlabel('Predicted')",
                "    ax.set_ylabel('True')",
                "    ax.tick_params(labelsize=7)",
                "    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')",
                "fig.suptitle('Row-normalised confusion matrices - identical class order and scale',",
                "             fontsize=13, fontweight='bold')",
                "plt.tight_layout()",
                "plt.savefig(EVAL_ROOT / 'confusion_matrices_comparison.png', dpi=150, bbox_inches='tight')",
                "plt.show()",
            ]
        ),
        make_code_cell(
            [
                "comparisons = [('baseline', 'baseline'), ('control', 'control')]",
                "low_rows = supports < LOW_SUPPORT",
                "",
                "fig, axes = plt.subplots(1, 2, figsize=(16, 7))",
                "for ax, (key, label) in zip(axes, comparisons):",
                "    diff_pp = (cm_norm - normalise(arms[key]['cm'])) * 100.0",
                "    diff_pp = np.where(low_rows[:, None], np.nan, diff_pp)",
                "    limit = np.nanmax(np.abs(diff_pp))",
                "    sns.heatmap(",
                "        diff_pp, annot=True, fmt='.0f', cmap='RdBu_r',",
                "        norm=TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit),",
                "        xticklabels=class_names, yticklabels=class_names,",
                "        ax=ax, linewidths=0.25, annot_kws={'fontsize': 6},",
                "        cbar_kws={'label': 'percentage points'}, mask=np.isnan(diff_pp),",
                "    )",
                '    ax.set_title(f"selected minus {label}\\n(excluded rows have fewer than {LOW_SUPPORT} samples)",',
                "                 fontsize=10)",
                "    ax.set_xlabel('Predicted')",
                "    ax.set_ylabel('True')",
                "    ax.tick_params(labelsize=7)",
                "    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')",
                "fig.suptitle('Difference in row proportion - selected model minus each arm',",
                "             fontsize=13, fontweight='bold')",
                "plt.tight_layout()",
                "plt.savefig(EVAL_ROOT / 'confusion_differences.png', dpi=150, bbox_inches='tight')",
                "plt.show()",
                "",
                "print(f'excluded rows ({int(low_rows.sum())}): '",
                "      f'{[class_names[i] for i in np.where(low_rows)[0]]}')",
            ]
        ),
        # ── 7. Headline comparison ───────────────────────────────────────────
        make_markdown_cell(
            [
                "## 7. Headline comparison\n",
                "\n",
                "The three arms on the same partition, measured by the same code. The difference\n",
                "columns are the selected model minus the other arm, so a positive value favours the\n",
                "selected model.\n",
                "\n",
                "The noise band was estimated from the run-to-run variability of macro F1, so it is\n",
                "applied to macro F1 and shown for that row only. The other metrics are reported as\n",
                "point differences with the sample sizes that qualify them, not as verdicts.",
            ]
        ),
        make_code_cell(
            [
                "def headline(arm):",
                "    m = arm['metrics']",
                "    return {",
                "        'Accuracy': m['overall_accuracy'],",
                "        'Macro F1': m['macro_f1'],",
                "        'Minimum class accuracy': m['min_class_accuracy'],",
                "    }",
                "",
                "",
                "scores = {name: headline(arm) for name, arm in arms.items()}",
                "comparison = pd.DataFrame({'Metric': list(scores['selected'])})",
                "for name in arms:",
                "    comparison[arms[name]['label']] = list(scores[name].values())",
                "",
                "sense_col = arms['selected']['label']",
                "for name in ('baseline', 'control'):",
                "    other = arms[name]['label']",
                "    comparison[f'diff vs {name}'] = comparison[sense_col] - comparison[other]",
                "    comparison[f'within macro-F1 band vs {name}'] = [",
                "        ('yes' if abs(delta) < NOISE_BAND else 'no') if metric == 'Macro F1' else 'not applicable'",
                "        for metric, delta in zip(comparison['Metric'], comparison[f'diff vs {name}'])",
                "    ]",
                "",
                "display(comparison.round(6))",
                "comparison.round(6).to_csv(EVAL_ROOT / 'comparison_table.csv', index=False)",
                "print('saved', (EVAL_ROOT / 'comparison_table.csv').relative_to(ROOT))",
            ]
        ),
        make_code_cell(
            [
                "fig, ax = plt.subplots(figsize=(11, 4.6))",
                "metrics_list = comparison['Metric'].tolist()",
                "x = np.arange(len(metrics_list))",
                "width = 0.26",
                "colours = {'selected': '#4c72b0', 'baseline': '#dd8452', 'control': '#55a868'}",
                "",
                "for offset, name in zip((-width, 0.0, width), arms):",
                "    values = [scores[name][m] for m in metrics_list]",
                "    bars = ax.bar(x + offset, values, width, label=arms[name]['label'], color=colours[name])",
                "    for xi, v in zip(x + offset, values):",
                "        ax.text(xi, v + 0.008, f'{v:.3f}', ha='center', fontsize=7.5)",
                "",
                "ax.set_xticks(x)",
                "ax.set_xticklabels(metrics_list)",
                "ax.set_ylim(0, 1.08)",
                "ax.set_ylabel('Proportion')",
                "ax.set_title('Headline metrics on the standard test partition', fontweight='bold')",
                "ax.legend(loc='lower right', fontsize=8)",
                "plt.tight_layout()",
                "plt.savefig(EVAL_ROOT / 'headline_comparison.png', dpi=150, bbox_inches='tight')",
                "plt.show()",
                "",
                "for metric in metrics_list:",
                "    row = comparison.loc[comparison['Metric'] == metric].iloc[0]",
                "    for name in ('baseline', 'control'):",
                "        line = f'{metric:<26} vs {name:<9} {row[f\"diff vs {name}\"]:+.4f}'",
                "        if metric == 'Macro F1':",
                "            state = row[f'within macro-F1 band vs {name}']",
                "            line += '  (inside band)' if state == 'yes' else '  (outside band)'",
                "        print(line)",
            ]
        ),
        # ── 8. Per-class gaps ────────────────────────────────────────────────
        make_markdown_cell(
            [
                "## 8. Per-class differences\n",
                "\n",
                "Per-class F1 for the selected model minus per-class F1 for each other arm, ordered by\n",
                "class index. Positive values favour the selected model.\n",
                "\n",
                "An aggregate difference can hide a split: a model can be better on some classes and\n",
                "worse on others by amounts larger than the aggregate. The per-class differences are\n",
                "not tested, and with seven samples in the smallest class a single prediction moves a\n",
                "per-class F1 substantially, so the charts describe where the arms disagree rather than\n",
                "rank them.",
            ]
        ),
        make_code_cell(
            [
                "fig, axes = plt.subplots(1, 2, figsize=(20, 5.6), sharey=True)",
                "tallies = {}",
                "",
                "for ax, (key, label) in zip(axes, comparisons):",
                "    other = arms[key]['per_class'].copy()",
                "    merged = per_class[['class_id', 'class_name', 'support', 'f1']].merge(",
                "        other[['class_id', 'f1']], on='class_id', suffixes=('_selected', '_other')",
                "    )",
                "    merged['gap'] = merged['f1_selected'] - merged['f1_other']",
                "    merged = merged.sort_values('class_id')",
                "",
                "    colours = ['#4c72b0' if g >= 0 else '#c44e52' for g in merged['gap']]",
                "    bars = ax.bar(merged['class_name'], merged['gap'], color=colours)",
                "    for bar, n in zip(bars, merged['support']):",
                "        offset = 0.006 if bar.get_height() >= 0 else -0.006",
                "        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset, f'{int(n)}',",
                "                ha='center', va='bottom' if offset > 0 else 'top', fontsize=6.5, color='0.35')",
                "    ax.axhline(0, color='0.2', linewidth=1)",
                "    ax.set_title(f'F1 difference - selected minus {label} (numbers under bars are supports)',",
                "                 fontsize=10)",
                "    ax.set_ylabel('F1 difference')",
                "    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)",
                "",
                "    tallies[key] = (int((merged['gap'] > 0).sum()), int((merged['gap'] < 0).sum()))",
                "",
                "fig.suptitle('Per-class F1 differences on the standard test partition',",
                "             fontsize=13, fontweight='bold')",
                "plt.tight_layout()",
                "plt.savefig(EVAL_ROOT / 'per_class_gap.png', dpi=150, bbox_inches='tight')",
                "plt.show()",
                "",
                "for key, (pos, neg) in tallies.items():",
                "    print(f'vs {key:<9}: {pos} classes favour the selected model, {neg} favour {key}')",
            ]
        ),
        # ── 9. Reading rule and summary ──────────────────────────────────────
        make_markdown_cell(
            [
                "## 9. Reading rule and summary\n",
                "\n",
                "The rule applied to this comparison was fixed before the run, and it is repeated here\n",
                "so that the summary cannot be read as a post-hoc choice:\n",
                "\n",
                "1. A macro-F1 difference smaller than the noise band is unresolved, in either direction.\n",
                "2. Per-class results are reported in full, for every class, whatever the aggregate says.\n",
                "3. A reversal of the validation ordering is recorded as a finding.\n",
                "4. A minimum-class difference is qualified by the support of the class that attains it.\n",
                "5. The three arms were fixed before the partition was read; no arm was added after\n",
                "   seeing a result, and no model was selected on the test partition.\n",
            ]
        ),
        make_code_cell(
            [
                "lines = []",
                "for name in ('selected', 'baseline', 'control'):",
                "    m = arms[name]['metrics']",
                "    lines.append(",
                "        f\"{arms[name]['label']}: accuracy {m['overall_accuracy']:.6f}, \"",
                "        f\"macro F1 {m['macro_f1']:.6f}, minimum class accuracy {m['min_class_accuracy']:.6f}.\"",
                "    )",
                "lines.append('')",
                "",
                "for name in ('baseline', 'control'):",
                "    row = comparison.loc[comparison['Metric'] == 'Macro F1'].iloc[0]",
                "    delta = row[f'diff vs {name}']",
                "    state = 'inside' if abs(delta) < NOISE_BAND else 'outside'",
                "    lines.append(f'Macro F1 difference vs {name}: {delta:+.4f} against a noise band of {NOISE_BAND:.3f} ({state}).')",
                "",
                "origin = comparison.loc[comparison['Metric'] == 'Minimum class accuracy'].iloc[0]",
                "for name in ('baseline', 'control'):",
                "    arm = arms[name]",
                "    cls = arm['per_class'].loc[arm['per_class']['accuracy'].idxmin()]",
                "    lines.append(",
                "        f\"The minimum for {name} is attained on '{cls['class_name']}' with n={int(cls['support'])}; \"",
                "        f\"the difference on that metric is {origin[f'diff vs {name}']:+.4f}.\"",
                "    )",
                "",
                "print('\\n'.join(lines))",
                "",
                "print()",
                "unresolved = [",
                "    name for name in ('baseline', 'control')",
                "    if abs(comparison.loc[comparison['Metric'] == 'Macro F1'].iloc[0][f'diff vs {name}']) < NOISE_BAND",
                "    ]",
                "if unresolved:",
                "    joined = ' and the '.join(unresolved)",
                "    noun, verb = ('difference', 'lies') if len(unresolved) == 1 else ('differences', 'lie')",
                "    print(f'VERDICT: the macro-F1 {noun} against the {joined} {verb} inside the noise band.')",
                "    print('         Those comparisons are UNRESOLVED. No superiority claim is made.')",
                "resolved = [name for name in ('baseline', 'control') if name not in unresolved]",
                "if resolved:",
                "    joined = ' and the '.join(resolved)",
                "    noun, verb = ('difference', 'lies') if len(resolved) == 1 else ('differences', 'lie')",
                "    print(f'VERDICT: the macro-F1 {noun} against the {joined} {verb} outside the noise band.')",
                "    print('         The direction is reported, subject to the single-seed limitation.')",
            ]
        ),
        make_markdown_cell(
            [
                "---\n",
                "\n",
                "**What this notebook does not do**\n",
                "\n",
                "- It does not select a model. All three checkpoints were frozen before the partition was read.\n",
                "- It does not test significance. The noise band is a threshold for interpretation, not a\n",
                "  confidence interval, and no p-value is computed here.\n",
                "- It does not report cross-scenario behaviour. No leave-one-scenario-out evaluation has\n",
                "  been run, so no robustness claim is available from this partition.\n",
                "- It does not attribute the difference to a single component. The control isolates the\n",
                "  three modules jointly; separating stereo from text guidance from enhancement requires\n",
                "  the ablation arms, which have not been trained.\n",
                "- It does not report latency. Inference speed was excluded from the study by decision, so\n",
                "  no efficiency claim is made.\n",
                "\n",
                "**Artefacts written by this notebook**\n",
                "\n",
                "| File | Content |\n",
                "|---|---|\n",
                "| `test_class_distribution.png` | Composition of the test partition by class |\n",
                "| `per_class_f1.png` | Per-class F1 for the selected model, support annotated |\n",
                "| `per_class_prf_heatmap.png` | Precision, recall, and F1 per class for the selected model |\n",
                "| `confusion_matrices_comparison.png` | The three arms on one figure, identical scale |\n",
                "| `confusion_differences.png` | Selected minus baseline, and selected minus control |\n",
                "| `headline_comparison.png` | Headline metrics for all three arms |\n",
                "| `per_class_gap.png` | Per-class F1 differences for both comparisons |\n",
                "| `comparison_table.csv` | The comparison table in machine-readable form |\n",
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


def main() -> None:
    out_path = ROOT / "notebooks" / "07_final_evaluation.ipynb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nb = build_notebook()
    for index, cell in enumerate(nb["cells"]):
        cell.setdefault("id", f"c{index:02d}")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Notebook written -> {out_path.relative_to(ROOT)}")
    print(
        "Execute with:\n"
        "  jupyter nbconvert --to notebook --execute --inplace notebooks/07_final_evaluation.ipynb"
    )


if __name__ == "__main__":
    main()
