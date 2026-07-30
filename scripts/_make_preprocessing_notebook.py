"""
scripts/_make_preprocessing_notebook.py
Creates notebooks/01_preprocessing.ipynb using pre-computed visualization outputs.
"""

from pathlib import Path

import nbformat as nbf

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
nb_path = Path("notebooks/01_preprocessing.ipynb")

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
        "# 01 — Data Preprocessing & Augmentation Pipeline QA\n\n"
        "**Phase 2.7 | SENSE Project**\n\n"
        "Visualizes raw vs. stereo-augmented image pairs and pixel intensity distributions\n"
        "before and after ImageNet normalization and on-the-fly augmentation.\n"
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
cells.append(
    nbf.v4.new_markdown_cell(
        "## 1. Raw vs. Augmented Stereo Pairs (Stereo-Consistent Parameter Sampling)"
    )
)

cells.append(
    nbf.v4.new_code_cell(
        "fig, ax = plt.subplots(figsize=(12, 10))\n"
        "img1 = mpimg.imread(str(PROCESSED_DIR / 'preprocessing_raw_vs_augmented.png'))\n"
        "ax.imshow(img1)\n"
        "ax.axis('off')\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
    )
)

# Section 2
cells.append(
    nbf.v4.new_markdown_cell("## 2. Pixel Intensity Distributions (Raw vs Standardized Tensors)")
)

cells.append(
    nbf.v4.new_code_cell(
        "fig, ax = plt.subplots(figsize=(15, 5))\n"
        "img2 = mpimg.imread(str(PROCESSED_DIR / 'preprocessing_rgb_histograms.png'))\n"
        "ax.imshow(img2)\n"
        "ax.axis('off')\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
    )
)

nb.cells = cells
nb_path.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(nb_path))
print(f"Notebook created at: {nb_path}")
