"""
scripts/_make_eda_notebook.py
Creates notebooks/00_data_exploration.ipynb using pre-computed EDA outputs.
Run once after scripts/run_eda.py has been executed.
"""

from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {
    "display_name": "Python 3 (.venv)",
    "language": "python",
    "name": "python3",
}
nb.metadata["language_info"] = {
    "name": "python",
    "version": "3.10.11",
}

cells = []

# ------------------------------------------------------------------
# Cell 0 — Title
# ------------------------------------------------------------------
cells.append(
    nbf.v4.new_markdown_cell(
        "# 00 — CADDY Dataset Exploratory Data Analysis\n\n"
        "**Phase 2.0 | SENSE Project**\n\n"
        "Surfaces class imbalance, scenario imbalance, and data-quality issues\n"
        "*before* any split is frozen. All findings are written to\n"
        "`data/processed/eda_findings.md`.\n\n"
        "Full computations also available as a script: `scripts/run_eda.py`.\n"
    )
)

# ------------------------------------------------------------------
# Cell 1 — Setup
# ------------------------------------------------------------------
cells.append(
    nbf.v4.new_code_cell(
        "import sys\n"
        "from pathlib import Path\n"
        "import matplotlib.pyplot as plt\n"
        "import matplotlib.image as mpimg\n"
        "import pandas as pd\n"
        "\n"
        "sys.path.insert(0, str(Path.cwd()))\n"
        "from src.config import Paths\n"
        "\n"
        "CADDY_ROOT    = Path(Paths.CADDY_ROOT)\n"
        "CSV_POSITIVES = Path(Paths.CADDY_CSV_POSITIVES)\n"
        "CSV_NEGATIVES = Path(Paths.CADDY_CSV_NEGATIVES)\n"
        "PROCESSED_DIR = Path('data/processed')\n"
        "\n"
        "df_pos = pd.read_csv(CSV_POSITIVES)\n"
        "df_neg = pd.read_csv(CSV_NEGATIVES)\n"
        "df_pos_real = df_pos[df_pos['synthetic'] == 0].copy()\n"
        "df_neg_real = df_neg[df_neg['synthetic'] == 0].copy()\n"
        "\n"
        "print('Real positive stereo pairs :', len(df_pos_real))\n"
        "print('Real negative stereo pairs :', len(df_neg_real))\n"
        "print('Total real stereo pairs    :', len(df_pos_real) + len(df_neg_real))\n"
    )
)

# ------------------------------------------------------------------
# Cell 2 — Per-class distribution
# ------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("## 1. Per-gesture-class sample distribution"))

cells.append(
    nbf.v4.new_code_cell(
        "gesture_counts = df_pos_real['label name'].value_counts().sort_index()\n"
        "print(gesture_counts.to_string())\n"
        "min_cls = gesture_counts.idxmin()\n"
        "max_cls = gesture_counts.idxmax()\n"
        "ratio = gesture_counts.max() / gesture_counts.min()\n"
        "print(f'\\nMIN: {gesture_counts.min()} ({min_cls})')\n"
        "print(f'MAX: {gesture_counts.max()} ({max_cls})')\n"
        "print(f'Imbalance ratio: {ratio:.1f}x')\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(13, 5))\n"
        "img = mpimg.imread(str(PROCESSED_DIR / 'eda_class_distribution.png'))\n"
        "ax.imshow(img)\n"
        "ax.axis('off')\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
    )
)

# ------------------------------------------------------------------
# Cell 3 — Per-scenario distribution
# ------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("## 2. Per-scenario sample distribution"))

cells.append(
    nbf.v4.new_code_cell(
        "scenario_pos = df_pos_real['scenario'].value_counts().sort_index()\n"
        "scenario_neg = df_neg_real['scenario'].value_counts().sort_index()\n"
        "scenario_df = pd.concat(\n"
        "    [scenario_pos.rename('positives'), scenario_neg.rename('negatives')], axis=1\n"
        ").fillna(0).astype(int)\n"
        "scenario_df['total'] = scenario_df['positives'] + scenario_df['negatives']\n"
        "print(scenario_df.to_string())\n"
        "scen_ratio = scenario_df['positives'].max() / scenario_df['positives'].min()\n"
        "print('Scenario imbalance ratio:', scen_ratio)\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(11, 5))\n"
        "img = mpimg.imread(str(PROCESSED_DIR / 'eda_scenario_distribution.png'))\n"
        "ax.imshow(img)\n"
        "ax.axis('off')\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
    )
)

# ------------------------------------------------------------------
# Cell 4 — Stereo spot-check
# ------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("## 3. Stereo pair visual spot-check (one per scenario)"))

cells.append(
    nbf.v4.new_code_cell(
        "fig, ax = plt.subplots(figsize=(10, 32))\n"
        "img = mpimg.imread(str(PROCESSED_DIR / 'eda_stereo_spot_check.png'))\n"
        "ax.imshow(img)\n"
        "ax.axis('off')\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
    )
)

# ------------------------------------------------------------------
# Cell 5 — Class per scenario
# ------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("## 4. Gesture class distribution per scenario"))

cells.append(
    nbf.v4.new_code_cell(
        "fig, ax = plt.subplots(figsize=(14, 6))\n"
        "img = mpimg.imread(str(PROCESSED_DIR / 'eda_class_per_scenario.png'))\n"
        "ax.imshow(img)\n"
        "ax.axis('off')\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
    )
)

# ------------------------------------------------------------------
# Cell 6 — Show findings
# ------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("## 5. EDA Findings Summary"))

cells.append(
    nbf.v4.new_code_cell(
        "findings_path = PROCESSED_DIR / 'eda_findings.md'\n"
        "print(findings_path.read_text(encoding='utf-8'))\n"
    )
)

nb.cells = cells

nb_path = Path("notebooks/00_data_exploration.ipynb")
nb_path.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(nb_path))
print(f"Notebook created: {nb_path}")
