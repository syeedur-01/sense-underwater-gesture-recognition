"""
scripts/_make_split_validation_notebook.py
Creates notebooks/02_dataset_split_validation.ipynb displaying split validation results.
"""

from pathlib import Path

import nbformat as nbf

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
nb_path = Path("notebooks/02_dataset_split_validation.ipynb")

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {
    "display_name": "Python 3 (sense-venv)",
    "language": "python",
    "name": "sense-venv",
}

cells = []

# Title
cells.append(
    nbf.v4.new_markdown_cell(
        "# 02 — Data Split Validation & Anti-Leakage Audit\n\n"
        "**Phase 2 Exit Deliverable | SENSE Project**\n\n"
        "Programmatically verifies zero sample leakage across standard 70/15/15 split\n"
        "and all 8 Leave-One-Scenario-Out (LOSO) cross-scenario folds.\n"
    )
)

# Setup
cells.append(
    nbf.v4.new_code_cell(
        "import os\n"
        f"PROJECT_ROOT = r'{PROJECT_ROOT}'\n"
        "os.chdir(PROJECT_ROOT)\n"
        "import sys\n"
        "from pathlib import Path\n"
        "import matplotlib.pyplot as plt\n"
        "import matplotlib.image as mpimg\n"
        "\n"
        "PROCESSED_DIR = Path('data/processed')\n"
        "print('Project Root:', PROJECT_ROOT)\n"
    )
)

# Section 1
cells.append(nbf.v4.new_markdown_cell("## 1. Split Validation & Anti-Leakage Audit Summary"))

cells.append(
    nbf.v4.new_code_cell(
        "report_path = PROCESSED_DIR / 'split_validation_report.md'\n"
        "print(report_path.read_text(encoding='utf-8'))\n"
    )
)

# Section 2
cells.append(nbf.v4.new_markdown_cell("## 2. LOSO 8-Fold Split Size Distribution"))

cells.append(
    nbf.v4.new_code_cell(
        "fig, ax = plt.subplots(figsize=(12, 5))\n"
        "img = mpimg.imread(str(PROCESSED_DIR / 'split_validation_summary.png'))\n"
        "ax.imshow(img)\n"
        "ax.axis('off')\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
    )
)

nb.cells = cells
nb_path.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(nb_path))
print(f"Notebook created at: {nb_path}")
