"""
scripts/_fix_notebook_path.py
Patches notebooks/00_data_exploration.ipynb to use an absolute project root
so it executes correctly under nbconvert regardless of kernel working directory.
"""

from pathlib import Path

import nbformat as nbf

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
nb_path = Path("notebooks/00_data_exploration.ipynb")

nb = nbf.read(str(nb_path), as_version=4)

# Fix setup cell (cell index 1) to chdir to project root before imports
old_src = nb.cells[1].source
insert = f"import os\n" f"PROJECT_ROOT = r'{PROJECT_ROOT}'\n" f"os.chdir(PROJECT_ROOT)\n"
nb.cells[1].source = insert + old_src

# Use the registered sense-venv kernel
nb.metadata["kernelspec"] = {
    "display_name": "Python 3 (sense-venv)",
    "language": "python",
    "name": "sense-venv",
}

nbf.write(nb, str(nb_path))
print(f"Notebook patched. PROJECT_ROOT = {PROJECT_ROOT}")
