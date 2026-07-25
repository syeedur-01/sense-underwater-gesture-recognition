# plan.md
## SENSE — Stereo-Enhanced kNowledge-guided Gesture Encoder
### Implementation Blueprint for Underwater Diver Gesture Recognition on CADDY

---

### How to use this document

This is the single source of truth for building SENSE end-to-end. Work through phases **in order** — each phase assumes all prior phases are complete and verified. Every task has an ID (`P{phase}.{task}`) so that future prompts can simply say **"Implement Phase 3 Task 2"** or **"Implement P3.2"** and the scope is unambiguous.

Each task specifies: **Objective**, **Depends on**, **Steps**, **Deliverable**, **Verification** (how to know it's actually done, not just "ran without crashing"). Do not mark a task complete until its Verification step passes.

**Golden rule for iterative development:** never build a component you cannot immediately test. Every phase ends with something runnable, inspectable, or measurable — not just code that "should work."

---

## Project Snapshot (context for any agent picking this up cold)

- **Goal:** Robust underwater diver gesture recognition on the CADDY stereo-vision dataset (15 gestures + no-gesture, 8 real-world scenarios).
- **Model:** SENSE — bi-channel EfficientNet-B4 stereo encoder → self-attention refinement → cross-modal attention fusion with CADDIAN class text embeddings → cosine-similarity classification head.
- **Baselines to reproduce/beat:** Yang et al. transfer learning (90.08% GoogleLeNet), DARE (95.87% overall / 92% min-class), VT-UHGR (best purely-visual beater, exact % not published).
- **Novelty type:** Combinational/integrative — first pipeline to combine (a) underwater image enhancement preprocessing, (b) explicit stereo depth fusion, and (c) semantic text guidance, evaluated with (d) systematic cross-scenario robustness testing on CADDY.
- **Hardware:** RTX 4060, 8GB VRAM. Mixed precision required for larger batch sizes.
- **Framework:** PyTorch.

---

## Phase Index

| Phase | Name | Depends On | Output |
|---|---|---|---|
| 0 | Bootstrap & Environment Setup | — | Reproducible dev environment |
| 1 | Data Acquisition & Auditing | 0 | Raw CADDY verified on disk |
| 2 | Data Pipeline & Preprocessing | 1 | PyTorch `Dataset`/`DataLoader`, scenario-aware splits |
| 3 | Enhancement Module Integration | 2 | FUnIE-GAN/WaterNet inference wrapper + cached enhanced dataset |
| 4 | Baseline Reproduction | 2 | Reproduced Yang et al. transfer-learning baseline (sanity check harness) |
| 5 | SENSE Model Construction | 3 | Full model, forward-pass verified, shape-tested |
| 6 | Training Infrastructure | 5 | Trainer, loss, scheduler, checkpointing, logging |
| 7 | Phase-Wise Model Training | 6 | Trained SENSE checkpoint (standard split) |
| 8 | Evaluation Framework | 7 | Metrics suite: accuracy, F1, confusion matrix, per-class, per-scenario |
| 9 | Cross-Scenario Robustness Study | 8 | Leave-scenario-out benchmark results |
| 10 | Ablation Studies | 7, 8 | Enhancement / stereo-vs-mono / text-guidance ablation results |
| 11 | Comparative Analysis vs DARE & VT-UHGR | 8, 9, 10 | Final comparison tables/figures |
| 12 | Results Packaging & Thesis Writeup Support | 11 | Figures, tables, exportable artifacts for report |
| 13 | Reproducibility & Handover | 12 | Final README, frozen configs, archived checkpoints |

---

## Phase 0 — Project Bootstrap & Environment Setup

### Objectives
- Prepare a fully reproducible development environment.
- Ensure any contributor (human or AI agent) can clone the repo and run the project from scratch with zero ambiguity.

### Depends on
Nothing. This is the starting point.

### Tasks

**P0.1 — Select Python version**
- Objective: Lock a single interpreter version to avoid dependency drift.
- Steps:
  1. Use **Python 3.10.x** (best compatibility window across PyTorch 2.x, timm/EfficientNet weights, sentence-transformers, and CUDA 12.x wheels as of this project's timeframe).
  2. Record the exact version in `.python-version` (for `pyenv`) and in `README.md`.
- Deliverable: `.python-version` file containing `3.10.13` (or latest 3.10 patch available).
- Verification: `python --version` inside the environment prints `3.10.x`.

**P0.2 — Create virtual environment**
- Objective: Isolate project dependencies.
- Steps:
  1. Create with: `python -m venv .venv`
  2. Activate: `source .venv/bin/activate` (Linux/Mac) or `.venv\Scripts\activate` (Windows).
  3. Upgrade pip/setuptools/wheel immediately after activation.
- Deliverable: `.venv/` directory (git-ignored).
- Verification: `which python` (or `where python` on Windows) points inside `.venv`.

**P0.3 — Define dependency management**
- Objective: Single, unambiguous dependency source.
- Steps:
  1. Use `requirements.txt` (simpler than `pyproject.toml` for this project size, and most compatible with quick agent-driven installs). Pin **major.minor** versions, not exact patch, to allow security patches:
     ```
     torch>=2.2,<2.3
     torchvision>=0.17,<0.18
     numpy>=1.26,<2.0
     pandas>=2.1
     scikit-learn>=1.4
     opencv-python>=4.9
     albumentations>=1.4
     pillow>=10.2
     matplotlib>=3.8
     seaborn>=0.13
     tqdm>=4.66
     pyyaml>=6.0
     tensorboard>=2.16
     sentence-transformers>=2.5
     timm>=0.9.16
     pytest>=8.0
     python-dotenv>=1.0
     ```
  2. Create a second file `requirements-dev.txt` for tooling only:
     ```
     black>=24.1
     ruff>=0.2
     pre-commit>=3.6
     ```
- Deliverable: `requirements.txt`, `requirements-dev.txt`.
- Verification: `pip install -r requirements.txt` completes with no errors on a clean venv.

**P0.4 — Install ML/DL frameworks with CUDA support**
- Objective: Working GPU-accelerated PyTorch on the RTX 4060.
- Steps:
  1. Identify installed NVIDIA driver's supported CUDA version (`nvidia-smi`, check "CUDA Version" in the header).
  2. Install PyTorch using the official CUDA 12.1 wheel index (RTX 4060 is Ada Lovelace, fully supported by CUDA 12.x):
     ```
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
     ```
  3. Do **not** install the CPU-only wheel by accident — verify in P0.6.
- Deliverable: PyTorch with CUDA installed inside `.venv`.
- Verification: see P0.6.

**P0.5 — Install supporting libraries**
- Objective: All non-DL-framework dependencies available.
- Steps:
  1. `pip install -r requirements.txt` (already covers NumPy, Pandas, scikit-learn, OpenCV, Albumentations, sentence-transformers, timm, etc.)
  2. `pip install -r requirements-dev.txt`
- Deliverable: Fully populated `.venv`.
- Verification: `pip list` shows all packages from both requirement files with no version conflicts (`pip check` returns no broken requirements).

**P0.6 — Configure GPU support and verify CUDA availability**
- Objective: Confirm PyTorch can actually see and use the RTX 4060.
- Steps:
  1. Create `scripts/check_env.py`:
     ```python
     import torch, platform, sys

     def main():
         print(f"Python: {sys.version}")
         print(f"Platform: {platform.platform()}")
         print(f"PyTorch: {torch.__version__}")
         print(f"CUDA available: {torch.cuda.is_available()}")
         if torch.cuda.is_available():
             print(f"CUDA device: {torch.cuda.get_device_name(0)}")
             print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
             print(f"VRAM (GB): {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}")
         else:
             print("WARNING: CUDA not available — training will run on CPU (not viable for this project).")

     if __name__ == "__main__":
         main()
     ```
  2. Run it: `python scripts/check_env.py`.
- Deliverable: `scripts/check_env.py`.
- Verification: Output shows `CUDA available: True`, device name containing "4060", and VRAM ≈ 8.0 GB.

**P0.7 — Configure environment variables**
- Objective: Centralize paths and secrets/config outside source code.
- Steps:
  1. Create `.env.example`:
     ```
     # Dataset paths
     CADDY_ROOT=./data/raw/caddy
     CADDY_CSV_POSITIVES=./data/raw/caddy/caddy_true_positives.csv
     CADDY_CSV_NEGATIVES=./data/raw/caddy/caddy_true_negatives.csv

     # Cache / derived data
     ENHANCED_CACHE_DIR=./data/processed/enhanced
     SPLITS_DIR=./data/processed/splits

     # Pretrained weights
     FUNIE_GAN_WEIGHTS=./weights/funie_gan/generator.pth
     WATERNET_WEIGHTS=./weights/waternet/waternet.pth

     # Experiment tracking
     RUNS_DIR=./runs
     CHECKPOINTS_DIR=./checkpoints

     # Reproducibility
     SEED=42

     # Hardware
     DEVICE=cuda
     ```
  2. Copy to `.env` locally (git-ignored) and fill in real values.
  3. Add `python-dotenv` loading utility in `src/config.py` (built out fully in P0.9).
- Deliverable: `.env.example` (committed), `.env` (local only, git-ignored).
- Verification: A script that imports `src/config.py` and prints resolved paths without raising `KeyError`.

**P0.8 — Create initial project directory structure**
- Objective: A stable, unambiguous folder layout every later phase writes into.
- Steps: create exactly this structure (empty `.gitkeep` files where needed to preserve empty dirs in git):
  ```
  sense-project/
  ├── .venv/                          # git-ignored
  ├── .env                            # git-ignored
  ├── .env.example
  ├── .python-version
  ├── .gitignore
  ├── requirements.txt
  ├── requirements-dev.txt
  ├── README.md
  ├── plan.md                         # this document
  ├── pyproject.toml                  # optional: black/ruff config only
  ├── data/
  │   ├── raw/
  │   │   └── caddy/                  # untouched original CADDY files
  │   └── processed/
  │       ├── enhanced/                # cached FUnIE-GAN/WaterNet outputs
  │       └── splits/                  # train/val/test + scenario split manifests
  ├── weights/
  │   ├── funie_gan/
  │   └── waternet/
  ├── src/
  │   ├── __init__.py
  │   ├── config.py
  │   ├── data/
  │   │   ├── __init__.py
  │   │   ├── caddy_dataset.py
  │   │   ├── transforms.py
  │   │   └── splits.py
  │   ├── enhancement/
  │   │   ├── __init__.py
  │   │   ├── funie_gan.py
  │   │   └── waternet.py
  │   ├── models/
  │   │   ├── __init__.py
  │   │   ├── visual_encoder.py
  │   │   ├── text_encoder.py
  │   │   ├── attention.py
  │   │   ├── fusion.py
  │   │   ├── classifier_head.py
  │   │   └── sense.py
  │   ├── training/
  │   │   ├── __init__.py
  │   │   ├── losses.py
  │   │   ├── trainer.py
  │   │   └── scheduler.py
  │   ├── evaluation/
  │   │   ├── __init__.py
  │   │   ├── metrics.py
  │   │   ├── cross_scenario.py
  │   │   └── ablations.py
  │   └── utils/
  │       ├── __init__.py
  │       ├── logging.py
  │       ├── seed.py
  │       └── checkpoint.py
  ├── scripts/
  │   ├── check_env.py
  │   ├── verify_dataset.py
  │   ├── cache_enhanced_images.py
  │   ├── train.py
  │   ├── evaluate.py
  │   ├── run_ablation.py
  │   └── run_cross_scenario.py
  ├── configs/
  │   ├── base.yaml
  │   ├── baseline_transfer_learning.yaml
  │   ├── sense_full.yaml
  │   ├── sense_no_enhancement.yaml
  │   ├── sense_mono.yaml
  │   └── sense_no_text.yaml
  ├── notebooks/
  │   └── 00_data_exploration.ipynb
  ├── tests/
  │   ├── __init__.py
  │   ├── test_data.py
  │   ├── test_models.py
  │   └── test_training_step.py
  ├── runs/                            # tensorboard logs, git-ignored
  └── checkpoints/                     # model weights, git-ignored
  ```
- Deliverable: Full directory tree committed (with `.gitkeep` in empty dirs).
- Verification: `find . -type d | sort` matches the structure above (minus git-ignored dirs' contents).

**P0.9 — Build `src/config.py` (centralized config loader)**
- Objective: One place all other code reads paths/settings from — never hardcode paths anywhere else.
- Steps:
  1. Implement using `python-dotenv` + `PyYAML`:
     ```python
     import os
     from pathlib import Path
     from dotenv import load_dotenv
     import yaml

     load_dotenv()

     class Paths:
         CADDY_ROOT = Path(os.getenv("CADDY_ROOT", "./data/raw/caddy"))
         CADDY_CSV_POSITIVES = Path(os.getenv("CADDY_CSV_POSITIVES"))
         CADDY_CSV_NEGATIVES = Path(os.getenv("CADDY_CSV_NEGATIVES"))
         ENHANCED_CACHE_DIR = Path(os.getenv("ENHANCED_CACHE_DIR", "./data/processed/enhanced"))
         SPLITS_DIR = Path(os.getenv("SPLITS_DIR", "./data/processed/splits"))
         FUNIE_GAN_WEIGHTS = Path(os.getenv("FUNIE_GAN_WEIGHTS"))
         WATERNET_WEIGHTS = Path(os.getenv("WATERNET_WEIGHTS"))
         RUNS_DIR = Path(os.getenv("RUNS_DIR", "./runs"))
         CHECKPOINTS_DIR = Path(os.getenv("CHECKPOINTS_DIR", "./checkpoints"))

     SEED = int(os.getenv("SEED", 42))
     DEVICE = os.getenv("DEVICE", "cuda")

     def load_yaml_config(path: str) -> dict:
         with open(path, "r") as f:
             return yaml.safe_load(f)
     ```
  2. Create `configs/base.yaml` with shared defaults (image size, num_classes=16, batch_size, etc.) that experiment-specific configs inherit/override.
- Deliverable: `src/config.py`, `configs/base.yaml`.
- Verification: `python -c "from src.config import Paths, SEED; print(Paths.CADDY_ROOT, SEED)"` runs without error.

**P0.10 — Configure Git and `.gitignore`**
- Objective: Keep the repo clean; never commit data, weights, or environments.
- Steps:
  1. `git init` (if not already).
  2. Create `.gitignore`:
     ```
     .venv/
     __pycache__/
     *.pyc
     .env
     data/raw/
     data/processed/
     weights/*.pth
     weights/**/*.pth
     runs/
     checkpoints/
     *.ipynb_checkpoints
     .pytest_cache/
     .ruff_cache/
     *.egg-info/
     .DS_Store
     ```
  3. Initial commit: `git add . && git commit -m "chore: project bootstrap (Phase 0)"`.
- Deliverable: `.gitignore`, initial commit.
- Verification: `git status` shows no data/weights/env files tracked; `git log` shows the bootstrap commit.

**P0.11 — Add linting, formatting, pre-commit hooks (recommended, not optional in practice — prevents agent-introduced style drift)**
- Objective: Consistent code style across human and AI contributions.
- Steps:
  1. `pyproject.toml`:
     ```toml
     [tool.black]
     line-length = 100

     [tool.ruff]
     line-length = 100
     select = ["E", "F", "I"]
     ```
  2. `.pre-commit-config.yaml`:
     ```yaml
     repos:
       - repo: https://github.com/psf/black
         rev: 24.1.1
         hooks:
           - id: black
       - repo: https://github.com/astral-sh/ruff-pre-commit
         rev: v0.2.1
         hooks:
           - id: ruff
     ```
  3. `pre-commit install`.
- Deliverable: `pyproject.toml`, `.pre-commit-config.yaml`, hooks installed.
- Verification: `pre-commit run --all-files` passes cleanly on an empty/near-empty repo.

**P0.12 — Create configuration files (experiment configs)**
- Objective: Every experiment variant (full SENSE, no-enhancement, mono, no-text, baseline) is defined declaratively — never as ad hoc code edits.
- Steps: create `configs/base.yaml` (already started in P0.9) plus:
  - `configs/baseline_transfer_learning.yaml` — Phase 4 baseline.
  - `configs/sense_full.yaml` — full pipeline, Phase 7.
  - `configs/sense_no_enhancement.yaml`, `configs/sense_mono.yaml`, `configs/sense_no_text.yaml` — Phase 10 ablations (created now as stubs, filled in later).
  - Each config specifies: `model.*`, `data.*`, `training.*`, `enhancement.enabled`, `stereo.enabled`, `text_guidance.enabled` flags so ablations are config-switches, not code forks.
- Deliverable: All 6 YAML configs present (stubs acceptable for later-phase ones at this point).
- Verification: `load_yaml_config()` successfully parses every file in `configs/`.

**P0.13 — Verify the environment with a simple test script**
- Objective: End-to-end smoke test that the whole toolchain works together.
- Steps:
  1. Create `tests/test_env_smoke.py`:
     ```python
     import torch

     def test_cuda_available():
         assert torch.cuda.is_available(), "CUDA must be available for this project"

     def test_basic_tensor_op_on_gpu():
         x = torch.randn(4, 3, 224, 224, device="cuda")
         y = torch.nn.Conv2d(3, 16, 3).cuda()(x)
         assert y.shape == (4, 16, 222, 222)

     def test_timm_model_loads():
         import timm
         model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=0)
         assert model is not None

     def test_sentence_transformer_loads():
         from sentence_transformers import SentenceTransformer
         model = SentenceTransformer("all-MiniLM-L6-v2")
         emb = model.encode(["stop", "go up"])
         assert emb.shape[0] == 2
     ```
  2. Run: `pytest tests/test_env_smoke.py -v`.
- Deliverable: Passing smoke test suite.
- Verification: All 4 tests pass (note: `pretrained=True` for EfficientNet and the sentence-transformer download require internet access on first run — confirm this succeeds once, then cache is local).

**P0.14 — Document setup instructions in README.md**
- Objective: Anyone (human or agent) can go from empty machine to verified environment by reading one file.
- Steps: Write `README.md` with sections: Project Overview, Requirements, Setup (step-by-step commands mirroring P0.1–P0.13), Directory Structure (link/copy of P0.8 tree), Running Tests, Common Issues (e.g., CUDA mismatch).
- Deliverable: `README.md`.
- Verification: A fresh clone + following README top-to-bottom results in `pytest tests/test_env_smoke.py` passing with no undocumented manual steps.

### Phase 0 Deliverables Checklist
- [ ] Working virtual environment (`.venv`, Python 3.10.x)
- [ ] `requirements.txt` + `requirements-dev.txt` installed cleanly
- [ ] `.env.example` (+ local `.env`)
- [ ] Full project directory structure created
- [ ] `src/config.py` functional
- [ ] Git initialized, `.gitignore` correct, bootstrap commit made
- [ ] Pre-commit hooks installed and passing
- [ ] All 6 experiment config stubs present and parseable
- [ ] `scripts/check_env.py` confirms CUDA + RTX 4060 detected
- [ ] `tests/test_env_smoke.py` — all tests passing
- [ ] `README.md` setup instructions verified by clean run-through

**Phase 0 exit criterion:** `pytest tests/` passes fully, `python scripts/check_env.py` shows CUDA + GPU detected, and a person unfamiliar with the project can set up the environment using only `README.md`.

---

## Phase 1 — Data Acquisition & Auditing

### Objectives
- Get the official CADDY dataset onto disk in a known-good state.
- Catalogue exactly what's inside it before writing any pipeline code against assumptions.

### Depends on
Phase 0 complete.

### Tasks

**P1.1 — Download official CADDY dataset**
- Steps: Download from the official CADDIAN site (not Kaggle mirror — see prior discussion on citation integrity): Complete dataset (2.5GB), CSV complete dataset (true positives, 20.4MB), CSV complete dataset (true negatives, 12.8MB), and Hands ROI images (1.5GB, optional but recommended). Place under `data/raw/caddy/` exactly as downloaded (do not rename files yet).
- Deliverable: Raw files present in `data/raw/caddy/`.
- Verification: `ls data/raw/caddy/` shows expected archive names; total size ≈ 4GB (2.5 + 0.02 + 0.013 + 1.5).

**P1.2 — Extract and inventory raw files**
- Steps: Extract archives in place. Write `scripts/verify_dataset.py` that walks the extracted tree and reports: total image count, file extensions present, folder naming pattern (scenario folders, left/right subfolders, etc.), and cross-checks image count against the two CSVs' row counts.
- Deliverable: `scripts/verify_dataset.py`, console report.
- Verification: Reported image count is consistent with the paper's stated ~9,191 positive + ~7,190 negative stereo pairs (allow small discrepancy, but flag anything more than a few percent off).

**P1.3 — Parse and validate CSV annotations**
- Steps: Load both CSVs with Pandas inside `scripts/verify_dataset.py`. Validate: no missing gesture labels, exactly 15 gesture classes + no-gesture in the negatives file, every referenced image filename actually exists on disk, scenario field present and has exactly 8 unique values.
- Deliverable: Validation report printed and saved to `data/processed/dataset_audit_report.txt`.
- Verification: Zero "missing file" errors; scenario count == 8; gesture class count == 15 (+1 negative class).

**P1.4 — Document dataset schema**
- Steps: Write `data/processed/DATASET_SCHEMA.md` describing: CSV column meanings, folder layout, left/right image pairing convention (naming pattern that links a left image to its right counterpart), scenario ID → scenario description mapping (e.g., which of the 8 are "clear water," "turbid," "low light," etc. — extract these from the CADDY paper/CSV metadata, do not guess).
- Deliverable: `data/processed/DATASET_SCHEMA.md`.
- Verification: A teammate/agent with no prior CADDY exposure can locate any image's label and scenario using only this document.

### Phase 1 Deliverables Checklist
- [ ] Raw CADDY dataset downloaded and extracted from official source
- [ ] `scripts/verify_dataset.py` runs clean with zero missing-file errors
- [ ] `data/processed/dataset_audit_report.txt` generated
- [ ] `data/processed/DATASET_SCHEMA.md` written and accurate

**Phase 1 exit criterion:** Running `python scripts/verify_dataset.py` produces a report with zero integrity errors, and `DATASET_SCHEMA.md` fully documents the data layout.

---

## Phase 2 — Data Pipeline & Preprocessing

### Objectives
- Build a PyTorch `Dataset`/`DataLoader` pipeline that yields stereo pairs + labels + scenario metadata.
- Build the two split strategies needed later: standard random 80/20 (for baseline comparison) and scenario-aware splits (for Phase 9's cross-scenario study) — both defined **now**, so no later phase invents its own ad hoc split.

### Depends on
Phase 1 complete.

### Tasks

**P2.1 — Define the canonical data manifest**
- Steps: Write `src/data/splits.py` → function `build_manifest()` that reads both CSVs and produces one unified Pandas DataFrame with columns: `sample_id, left_path, right_path, gesture_label, gesture_id, scenario_id, is_negative`. Save as `data/processed/manifest.csv`.
- Deliverable: `data/processed/manifest.csv`, `build_manifest()` function.
- Verification: `len(manifest) == total image count from P1.2`; no null values in any required column.

**P2.2 — Implement standard random split (80/20, stratified)**
- Steps: In `src/data/splits.py`, add `make_standard_split(manifest, test_size=0.2, seed=SEED)` using `sklearn.model_selection.train_test_split` with `stratify=manifest['gesture_id']`. Save indices to `data/processed/splits/standard_train.csv` / `standard_test.csv`.
- Deliverable: Two split manifest CSVs.
- Verification: Class distribution (per gesture) in train and test sets differ by no more than ~2% proportionally (stratification sanity check, printed and asserted in a test).

**P2.3 — Implement scenario-aware leave-one-out splits**
- Steps: Add `make_scenario_splits(manifest)` producing 8 splits, each holding out one scenario entirely for testing and training on the remaining 7. Save as `data/processed/splits/scenario_holdout_{i}_train.csv` / `_test.csv` for `i` in 1..8.
- Deliverable: 16 CSV files (8 train/test pairs).
- Verification: For each holdout split, `test` set contains only the held-out scenario, `train` contains zero samples from it (assert set intersection is empty).

**P2.4 — Implement `CaddyStereoDataset` (PyTorch `Dataset`)**
- Steps: `src/data/caddy_dataset.py`:
  ```python
  class CaddyStereoDataset(torch.utils.data.Dataset):
      def __init__(self, manifest_csv, transform=None, enhanced_dir=None):
          ...
      def __len__(self): ...
      def __getitem__(self, idx):
          # returns dict: {"left": tensor, "right": tensor, "label": int,
          #                "scenario_id": int, "sample_id": str}
          ...
  ```
  - If `enhanced_dir` is provided and populated (post-Phase 3), load enhanced images from there instead of raw; otherwise load raw and apply only resize/normalize.
- Deliverable: `src/data/caddy_dataset.py`.
- Verification: `tests/test_data.py::test_dataset_getitem_shapes` — for a sample batch, `left`/`right` tensors are shape `(3, H, W)`, `label` is a valid int in `[0, 15]`.

**P2.5 — Implement transforms/augmentation pipeline**
- Steps: `src/data/transforms.py` using Albumentations. Two pipelines:
  - `train_transforms()`: resize → random horizontal flip → brightness/contrast jitter → Gaussian blur (p=0.3) → random crop+resize → normalize (ImageNet mean/std, since EfficientNet backbone is ImageNet-pretrained).
  - `eval_transforms()`: resize → normalize only (deterministic).
  - **Critical constraint:** any spatial augmentation (flip, crop) must be applied **identically** to left and right images in the same sample (pass both through the same Albumentations `Compose` call with a shared random seed per sample, or use Albumentations' multi-image API) — otherwise stereo correspondence breaks.
- Deliverable: `src/data/transforms.py`.
- Verification: `tests/test_data.py::test_stereo_augmentation_consistency` — apply a random flip transform to a synthetic left/right pair with identical content, confirm both come out flipped identically (not independently randomized).

**P2.6 — Implement `DataLoader` factory with correct batching**
- Steps: `src/data/caddy_dataset.py` → `get_dataloader(split_csv, batch_size, shuffle, num_workers, transform)`. Default `num_workers=4`, `pin_memory=True` (CUDA).
- Deliverable: `get_dataloader()` function.
- Verification: Iterate one batch, confirm `batch["left"].shape == (batch_size, 3, H, W)`, no exceptions, completes in reasonable time (<5s for first batch on cached data).

**P2.7 — Data exploration notebook**
- Steps: `notebooks/00_data_exploration.ipynb` — class distribution bar chart, scenario distribution bar chart, sample stereo pairs visualized (left|right side by side) for each of the 15 gesture classes, visual inspection of image quality across scenarios (spot the turbid/low-light ones).
- Deliverable: Executed notebook with outputs saved.
- Verification: Notebook runs top-to-bottom without errors; class imbalance (if any) is now documented, not assumed.

### Phase 2 Deliverables Checklist
- [ ] `data/processed/manifest.csv` built and validated
- [ ] Standard 80/20 stratified split created
- [ ] 8 scenario-holdout splits created
- [ ] `CaddyStereoDataset` implemented and unit-tested
- [ ] Stereo-consistent augmentation pipeline implemented and unit-tested
- [ ] `DataLoader` factory working
- [ ] Data exploration notebook completed

**Phase 2 exit criterion:** `pytest tests/test_data.py` passes; a `DataLoader` built from `configs/base.yaml` settings yields correctly shaped, correctly paired stereo batches.

---

## Phase 3 — Enhancement Module Integration

### Objectives
- Wrap FUnIE-GAN (primary) and WaterNet (ablation comparator) as inference-only modules.
- Pre-compute and cache enhanced versions of every image so training doesn't re-run GAN inference every epoch (critical for RTX 4060 throughput).

### Depends on
Phase 2 complete.

### Tasks

**P3.1 — Acquire pretrained FUnIE-GAN weights**
- Steps: Obtain the official pretrained FUnIE-GAN generator weights (from the original authors' released checkpoint). Place at `weights/funie_gan/generator.pth`. Document exact source URL/commit in `weights/funie_gan/SOURCE.md`.
- Deliverable: Weight file + source documentation.
- Verification: File loads via `torch.load()` without error; documented source is traceable/citable in the thesis.

**P3.2 — Acquire pretrained WaterNet weights**
- Steps: Same as P3.1 for WaterNet, at `weights/waternet/waternet.pth`, with `weights/waternet/SOURCE.md`.
- Deliverable: Weight file + source documentation.
- Verification: Loads without error; source documented.

**P3.3 — Implement `FUnIEGANEnhancer` wrapper**
- Steps: `src/enhancement/funie_gan.py`:
  ```python
  class FUnIEGANEnhancer:
      def __init__(self, weights_path, device="cuda"):
          ...
      @torch.no_grad()
      def enhance(self, image_tensor: torch.Tensor) -> torch.Tensor:
          # image_tensor: (B, 3, H, W) in [0,1] or [-1,1] per model's expected range
          # returns enhanced tensor, same shape, same range convention documented
          ...
  ```
  Include the correct generator architecture definition matching the pretrained weights (do not guess layer shapes — instantiate exactly per the original FUnIE-GAN architecture).
- Deliverable: `src/enhancement/funie_gan.py`.
- Verification: `tests/test_models.py::test_funie_gan_forward` — feed a random `(1,3,256,256)` tensor, confirm output shape matches input shape and values are in a valid image range after any needed denormalization.

**P3.4 — Implement `WaterNetEnhancer` wrapper**
- Steps: Same pattern as P3.3 in `src/enhancement/waternet.py`. Note WaterNet's gated fusion architecture requires the *color-corrected variants* as additional inputs per the original paper — implement the exact required preprocessing (white-balance / histogram-equalized variants) inside this wrapper so callers just pass a raw image.
- Deliverable: `src/enhancement/waternet.py`.
- Verification: `tests/test_models.py::test_waternet_forward` passes with correct output shape.

**P3.5 — Build enhancement caching script**
- Objective: Never run GAN inference inside the training loop — cache once, reuse always.
- Steps: `scripts/cache_enhanced_images.py`:
  - CLI args: `--method {funie_gan,waternet,none}`.
  - For every image in `manifest.csv`, run through the selected enhancer, save to `data/processed/enhanced/{method}/{sample_id}_L.png` and `_R.png`.
  - Log progress with `tqdm`, resumable (skip already-cached files).
- Deliverable: `scripts/cache_enhanced_images.py`.
- Verification: Run for a small subset (e.g., 50 samples) first — confirm output images visually look enhanced (less blue/green cast, more contrast) by eye-checking a few in the data exploration notebook. Then run full dataset for both methods.

**P3.6 — Visual QA of enhancement outputs**
- Steps: Extend `notebooks/00_data_exploration.ipynb` (or new `01_enhancement_qa.ipynb`) with side-by-side raw vs FUnIE-GAN vs WaterNet comparisons across at least one sample from each of the 8 scenarios.
- Deliverable: QA notebook section with visual grid.
- Verification: Visual confirmation that enhancement is doing *something* sensible (not producing artifacts, black frames, or NaNs) — a human review checkpoint, not purely automated.

**P3.7 — Wire enhancement into `CaddyStereoDataset`**
- Steps: Confirm the `enhanced_dir` parameter added in P2.4 correctly points to `data/processed/enhanced/{method}/` and loads cached files. Add a `configs/*.yaml` field `data.enhancement_method: funie_gan | waternet | none` that controls this at the config level (this is what makes Phase 10's enhancement ablation a one-line config change).
- Deliverable: Updated dataset wiring, config field documented.
- Verification: `tests/test_data.py::test_dataset_loads_enhanced_variant` — instantiate dataset with `enhancement_method="funie_gan"`, confirm it loads from the cache dir, not raw dir.

### Phase 3 Deliverables Checklist
- [ ] FUnIE-GAN weights acquired, documented, wrapper implemented and tested
- [ ] WaterNet weights acquired, documented, wrapper implemented and tested
- [ ] Full dataset cached for `none`, `funie_gan`, and `waternet` variants
- [ ] Visual QA performed and documented
- [ ] Dataset class correctly switches enhancement source via config

**Phase 3 exit criterion:** All three cached image sets exist on disk; `CaddyStereoDataset` can load any of the three via a config flag; visual QA notebook confirms enhancement outputs are sane.

---

## Phase 4 — Baseline Reproduction

### Objectives
- Before building anything novel, reproduce a known result (Yang et al. transfer-learning baseline) on your exact data pipeline. This validates the entire data pipeline (Phases 1–2) is correct — if you can't hit a reasonable accuracy on a known-simple baseline, something upstream is broken, and you need to know that *before* debugging the much more complex SENSE model.

### Depends on
Phase 2 complete. (Does not require Phase 3 — this baseline uses raw images, matching the original paper's setup.)

### Tasks

**P4.1 — Implement simple single-image transfer-learning classifier**
- Steps: `src/models/visual_encoder.py` → add a minimal `SingleImageClassifier` (ResNet18 or GoogleLeNet/Inception-v1 backbone via `timm`, pretrained on ImageNet, replace final FC layer with 16-way head). This deliberately mirrors Yang et al.'s setup — single image, no stereo, no enhancement, no text.
- Deliverable: `SingleImageClassifier` class.
- Verification: Forward pass on a batch of shape `(B,3,224,224)` returns `(B,16)` logits.

**P4.2 — Write minimal training loop for baseline**
- Steps: `scripts/train.py` should support a `--config configs/baseline_transfer_learning.yaml` mode that: loads standard 80/20 split (P2.2), uses only the **left** image, trains `SingleImageClassifier` with cross-entropy loss, Adam optimizer, for a fixed small number of epochs (e.g., 15–20 — this is a sanity baseline, not the final model).
- Deliverable: Working `--config configs/baseline_transfer_learning.yaml` path through `train.py`.
- Verification: Loss decreases monotonically-ish over epochs (plot or printed log); training completes without OOM on RTX 4060.

**P4.3 — Evaluate baseline and compare to literature**
- Steps: Run held-out test evaluation, report overall accuracy.
- Deliverable: `data/processed/baseline_results.md` with achieved accuracy vs. Yang et al.'s reported 90.08% (GoogleLeNet) / 88.59% (ResNet).
- Verification: Achieved accuracy is in a plausible range (does not need to exactly match, but should be within the same ballpark, e.g., 80%+ — if it's near-random ~6%, the data pipeline has a bug that must be fixed before proceeding to any later phase).

### Phase 4 Deliverables Checklist
- [ ] `SingleImageClassifier` implemented
- [ ] Baseline training run completed
- [ ] Baseline evaluation report written, compared against literature values

**Phase 4 exit criterion:** Baseline accuracy is plausible (not near-random), confirming the data pipeline is trustworthy before building SENSE on top of it. **Do not proceed to Phase 5 until this checkpoint is genuinely satisfied.**

---

## Phase 5 — SENSE Model Construction

### Objectives
- Build each SENSE component in isolation, unit-test its shape/behavior, then assemble into the full model — never write the full model in one pass.

### Depends on
Phase 3 complete (enhancement caching available), Phase 4 complete (pipeline trust established).

### Tasks

**P5.1 — Implement bi-channel visual encoder**
- Steps: `src/models/visual_encoder.py` → `BiChannelEncoder`:
  ```python
  class BiChannelEncoder(nn.Module):
      def __init__(self, backbone_name="efficientnet_b4", pretrained=True,
                   shared_weights=False, out_dim=512):
          # if shared_weights=False: two independent timm EfficientNet-B4 instances
          # if shared_weights=True: one instance, called twice
          ...
      def forward(self, left, right):
          # returns fused (B, out_dim) tensor after concat + linear projection
          ...
  ```
  - `shared_weights` flag exists from the start (needed for the weight-sharing ablation mentioned in earlier discussion).
- Deliverable: `BiChannelEncoder` class.
- Verification: `tests/test_models.py::test_bichannel_encoder_shape` — input `(B,3,224,224)` × 2 → output `(B,512)`.

**P5.2 — Implement visual self-attention module**
- Steps: `src/models/attention.py` → `SelfAttentionBlock` wrapping `nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)`. Since input is a single 512-dim vector per sample (not a sequence), reshape appropriately — treat the 512-dim vector as an 8-token sequence of 64-dim chunks, or use a single-token self-attention with a learned residual (choose the single-token + residual + LayerNorm formulation for simplicity and document this choice explicitly in the code docstring, since "self-attention over a single vector" is a design decision that needs justification in the thesis, not just code).
- Deliverable: `SelfAttentionBlock` class.
- Verification: `tests/test_models.py::test_self_attention_shape` — input `(B,512)` → output `(B,512)`, no shape drift.

**P5.3 — Implement text encoder + fixed embedding cache**
- Steps: `src/models/text_encoder.py`:
  ```python
  class GestureTextEncoder:
      def __init__(self, class_names: list[str], method="sentence_bert", embed_dim=512):
          # method options: "word2vec" | "sentence_bert" | "bert"
          ...
      def get_embeddings(self) -> torch.Tensor:
          # returns (num_classes, embed_dim), computed ONCE, cached in memory
          ...
  ```
  - Class names list must exactly match the 16 CADDIAN classes documented in `DATASET_SCHEMA.md` (P1.4) — pull from there, don't hardcode a second source of truth.
  - If embedding dim from the chosen text model ≠ 512, add a learned linear projection layer to bring it to 512 (must match visual feature dim for fusion).
- Deliverable: `GestureTextEncoder` class.
- Verification: `tests/test_models.py::test_text_encoder_shape` — `get_embeddings()` returns `(16, 512)`; calling twice returns identical values (confirms caching, not re-randomized).

**P5.4 — Implement cross-modal attention fusion**
- Steps: `src/models/fusion.py` → `CrossModalFusion`:
  ```python
  class CrossModalFusion(nn.Module):
      def __init__(self, embed_dim=512, num_heads=8):
          # two nn.MultiheadAttention modules: visual->text and text->visual
          ...
      def forward(self, visual_feat, text_embeddings):
          # visual_feat: (B, 512) ; text_embeddings: (16, 512), broadcast across batch
          # returns joint_features: (B, 512)
          ...
  ```
- Deliverable: `CrossModalFusion` class.
- Verification: `tests/test_models.py::test_cross_modal_fusion_shape` — `(B,512)` + `(16,512)` → `(B,512)` output, gradients flow (test `.backward()` doesn't error).

**P5.5 — Implement cosine-similarity classification head**
- Steps: `src/models/classifier_head.py` → `CosineSimilarityHead`:
  ```python
  class CosineSimilarityHead(nn.Module):
      def __init__(self, temperature=0.07):
          ...
      def forward(self, joint_features, text_embeddings):
          # returns similarities (B, num_classes) — used as logits for CE loss
          ...
  ```
  - `temperature` is a learnable or fixed scalar (start fixed, make learnable if validation performance suggests it in Phase 7).
- Deliverable: `CosineSimilarityHead` class.
- Verification: `tests/test_models.py::test_classifier_head_shape` — output `(B,16)`; row-wise softmax sums to 1.

**P5.6 — Assemble full `SENSE` model**
- Steps: `src/models/sense.py`:
  ```python
  class SENSE(nn.Module):
      def __init__(self, config):
          # wires together: BiChannelEncoder, SelfAttentionBlock,
          # GestureTextEncoder, CrossModalFusion, CosineSimilarityHead
          # respects config flags: stereo.enabled, text_guidance.enabled
          # (mono mode: right branch either zeroed or skipped;
          #  no-text mode: skip fusion, classify visual features directly
          #  via a fallback linear classifier head — see P5.7)
          ...
      def forward(self, left, right):
          # returns logits (B, 16), and optionally intermediate features for inspection
          ...
  ```
- Deliverable: `SENSE` class, fully wired.
- Verification: `tests/test_models.py::test_sense_full_forward` — full forward pass on a real batch from `DataLoader` (Phase 2/3 pipeline), output shape `(B,16)`, no NaNs.

**P5.7 — Implement ablation-mode switches inside `SENSE`**
- Objective: Phase 10's ablations must be config flags, not separate model classes, to guarantee fair comparison (identical training code path).
- Steps: Add to `SENSE.__init__`:
  - `stereo.enabled: bool` — if `False`, only the left branch of `BiChannelEncoder` is used (mono mode); right branch is not instantiated (not just masked — this matters for the "does stereo help" ablation to be honest about parameter count too).
  - `text_guidance.enabled: bool` — if `False`, skip `CrossModalFusion` and `CosineSimilarityHead` entirely; instead attach a plain `nn.Linear(512, 16)` classifier directly on the (optionally self-attended) visual features.
  - `enhancement` is **not** a model-level flag — it's handled entirely at the data level (Phase 3), so the model always receives whatever images the dataset gives it. Document this clearly so nobody duplicates the enhancement ablation logic inside the model by mistake.
- Deliverable: Updated `SENSE` class with working mode switches.
- Verification: `tests/test_models.py::test_sense_ablation_modes` — instantiate with `stereo.enabled=False`, confirm right-branch parameters don't exist (`sum(p.numel() for p in model.parameters())` is measurably smaller); instantiate with `text_guidance.enabled=False`, confirm forward pass still returns `(B,16)` via the fallback linear head.

**P5.8 — Parameter count and memory budget check**
- Steps: Script or test that prints total parameter count and estimates peak VRAM for a forward+backward pass at the intended batch size, using `torch.cuda.max_memory_allocated()` after one training step.
- Deliverable: Printed report, saved to `data/processed/model_memory_report.txt`.
- Verification: Peak VRAM usage at chosen batch size stays under ~7GB (leaving headroom on the 8GB RTX 4060); if it doesn't, reduce batch size or enable gradient checkpointing before Phase 6.

### Phase 5 Deliverables Checklist
- [ ] `BiChannelEncoder` implemented, tested, supports weight-sharing flag
- [ ] `SelfAttentionBlock` implemented, tested
- [ ] `GestureTextEncoder` implemented, tested, caching confirmed
- [ ] `CrossModalFusion` implemented, tested, gradients confirmed flowing
- [ ] `CosineSimilarityHead` implemented, tested
- [ ] Full `SENSE` model assembled and forward-pass verified on real data
- [ ] All ablation-mode switches implemented and tested
- [ ] Memory budget verified safe for RTX 4060

**Phase 5 exit criterion:** `pytest tests/test_models.py` passes in full; `SENSE` runs a forward+backward pass on a real batch from the Phase 2/3 pipeline within VRAM budget, in all four mode combinations (full / no-stereo / no-text / no-stereo-no-text).

---

## Phase 6 — Training Infrastructure

### Objectives
- Build the reusable trainer, loss, scheduler, checkpointing, and logging systems once — every later training run (baseline already done in P4, but SENSE variants in Phases 7/9/10) reuses this exact infrastructure.

### Depends on
Phase 5 complete.

### Tasks

**P6.1 — Implement combined loss function**
- Steps: `src/training/losses.py`:
  ```python
  class SenseLoss(nn.Module):
      def __init__(self, alpha=0.7, beta=0.3):
          # alpha: cross-entropy weight, beta: contrastive weight
          ...
      def forward(self, logits, labels, joint_features=None, text_embeddings=None):
          ce = F.cross_entropy(logits, labels)
          if joint_features is not None and text_embeddings is not None:
              contrastive = self._contrastive_loss(joint_features, text_embeddings, labels)
              return self.alpha * ce + self.beta * contrastive, {"ce": ce.item(), "contrastive": contrastive.item()}
          return ce, {"ce": ce.item()}
      def _contrastive_loss(self, joint_features, text_embeddings, labels):
          # InfoNCE-style: pull matching image-text pairs together,
          # push non-matching pairs apart
          ...
  ```
  - When `text_guidance.enabled=False` (Phase 5.7 fallback mode), only CE loss applies — handle this gracefully (contrastive term simply omitted, not zero-multiplied wastefully).
- Deliverable: `src/training/losses.py`.
- Verification: `tests/test_training_step.py::test_loss_computation` — loss is a positive scalar, finite, `.backward()` succeeds.

**P6.2 — Implement optimizer + cosine annealing scheduler factory**
- Steps: `src/training/scheduler.py` → `build_optimizer_and_scheduler(model, config)` returning AdamW optimizer + `torch.optim.lr_scheduler.CosineAnnealingLR`, with parameters (base LR, T_max, weight decay) read from config.
- Deliverable: `src/training/scheduler.py`.
- Verification: Optimizer step + scheduler step run for a few dummy iterations without error; LR decreases as expected (assert `scheduler.get_last_lr()` trends downward).

**P6.3 — Implement 3-phase progressive unfreezing controller**
- Steps: `src/training/trainer.py` — logic that, given current epoch and config-defined phase boundaries (default: Phase 1 epochs 1–10 frozen backbone, Phase 2 epochs 11–30 unfreeze last 2 EfficientNet blocks, Phase 3 epochs 31–50 full fine-tune), sets `requires_grad` appropriately on `BiChannelEncoder` submodules at the correct epoch transitions, and rebuilds the optimizer's parameter groups accordingly (with a lower LR for newly-unfrozen phases, per config).
- Deliverable: `PhaseController` class or equivalent function inside `trainer.py`.
- Verification: `tests/test_training_step.py::test_progressive_unfreezing` — at epoch 1, backbone params have `requires_grad=False`; at epoch 11, last-2-block params flip to `True`; at epoch 31, all params are `True`.

**P6.4 — Implement checkpointing utility**
- Steps: `src/utils/checkpoint.py` → `save_checkpoint(model, optimizer, scheduler, epoch, metrics, path)` and `load_checkpoint(path, model, optimizer=None, scheduler=None)`. Always save: model state dict, optimizer state, scheduler state, epoch number, validation metrics dict, and the exact config used (serialize the YAML alongside — critical for reproducibility across ablation runs).
- Deliverable: `src/utils/checkpoint.py`.
- Verification: `tests/test_training_step.py::test_checkpoint_roundtrip` — save then load a checkpoint, confirm model weights are bit-identical before/after.

**P6.5 — Implement TensorBoard logging**
- Steps: `src/utils/logging.py` → thin wrapper around `torch.utils.tensorboard.SummaryWriter` logging: train/val loss (total + CE + contrastive components separately), train/val accuracy, learning rate, epoch timing, and (from epoch 1) a fixed validation batch's per-class prediction confidence for qualitative tracking.
- Deliverable: `src/utils/logging.py`.
- Verification: After a short dummy run, `runs/` contains a TensorBoard event file; `tensorboard --logdir runs/` displays populated scalar charts.

**P6.6 — Implement seed control for reproducibility**
- Steps: `src/utils/seed.py` → `set_seed(seed)` setting Python `random`, NumPy, `torch.manual_seed`, `torch.cuda.manual_seed_all`, and `torch.backends.cudnn.deterministic=True` (accept the throughput cost for reproducibility, since experiment comparability matters more than raw speed here).
- Deliverable: `src/utils/seed.py`.
- Verification: Running the same short training config twice with the same seed produces identical loss values at epoch 1 (assert equality on the logged first-epoch loss).

**P6.7 — Assemble the full `Trainer` class**
- Steps: `src/training/trainer.py` → `Trainer` class tying together: dataloaders, model, loss, optimizer/scheduler, phase controller, checkpointing, logging, early stopping (based on validation accuracy plateau, patience configurable). Expose a single `.fit(num_epochs)` method and a `.validate()` method usable standalone (needed by Phase 8's evaluation framework).
- Deliverable: `src/training/trainer.py`.
- Verification: `tests/test_training_step.py::test_trainer_single_epoch` — run `Trainer.fit(num_epochs=1)` on a tiny synthetic/subset dataset, confirm it completes, logs something, saves a checkpoint, and loss is finite.

**P6.8 — Wire `scripts/train.py` CLI**
- Steps: `scripts/train.py --config configs/{name}.yaml [--resume checkpoints/{path}]`. Loads config, builds dataset/model/trainer per config contents, calls `.fit()`.
- Deliverable: `scripts/train.py`.
- Verification: `python scripts/train.py --config configs/sense_full.yaml` (with epochs temporarily overridden to 1 for a smoke test) runs end-to-end without error.

### Phase 6 Deliverables Checklist
- [ ] Combined CE + contrastive loss implemented and tested
- [ ] Optimizer/scheduler factory implemented
- [ ] 3-phase progressive unfreezing controller implemented and tested
- [ ] Checkpointing save/load round-trip verified
- [ ] TensorBoard logging verified populated
- [ ] Seed control verified reproducible
- [ ] Full `Trainer` class assembled and one-epoch smoke test passes
- [ ] `scripts/train.py` CLI functional end-to-end

**Phase 6 exit criterion:** A 1-epoch smoke-test run of `scripts/train.py --config configs/sense_full.yaml` completes successfully, produces a checkpoint, and populates TensorBoard logs.

---

## Phase 7 — Phase-Wise Model Training (Standard Split)

### Objectives
- Train the full SENSE model to convergence on the standard 80/20 split, producing the primary checkpoint used for headline results and all comparisons against DARE/VT-UHGR.

### Depends on
Phase 6 complete.

### Tasks

**P7.1 — Finalize `configs/sense_full.yaml`**
- Steps: Set final hyperparameters: batch size (start at 16, adjust per P5.8 memory budget), 50 total epochs per the 3-phase schedule, `alpha=0.7, beta=0.3` initial loss weights, `enhancement_method: funie_gan`, `stereo.enabled: true`, `text_guidance.enabled: true`, standard split paths.
- Deliverable: Finalized `configs/sense_full.yaml`.
- Verification: Config loads and every referenced path exists.

**P7.2 — Run full training**
- Steps: `python scripts/train.py --config configs/sense_full.yaml`. Monitor via TensorBoard during the run. Expect multi-hour runtime on RTX 4060 — plan for this, checkpoint frequently (every epoch minimum) so a crash doesn't lose progress.
- Deliverable: `checkpoints/sense_full_best.pth`, `checkpoints/sense_full_last.pth`.
- Verification: Training completes all 50 epochs (or triggers early stopping sensibly, not prematurely at epoch 3); validation accuracy trend is visible and generally improving in TensorBoard.

**P7.3 — Hyperparameter sanity sweep (light touch)**
- Steps: If P7.2's validation accuracy is clearly underperforming expectations (e.g., far below the 90% baseline range), try 2–3 targeted variations: adjust `alpha`/`beta` loss weighting, adjust base learning rate, adjust batch size. Do **not** do an exhaustive grid search — this is a thesis project with limited compute, so 2–3 deliberate, documented attempts is appropriate.
- Deliverable: `data/processed/hparam_sweep_notes.md` documenting what was tried and why.
- Verification: Final chosen config is justified in writing, not just "the one that happened to run last."

**P7.4 — Select and freeze final standard-split checkpoint**
- Steps: Based on validation performance, pick the best checkpoint. Rename/copy to `checkpoints/sense_final_standard_split.pth`. This is the checkpoint referenced in Phase 8/11 headline comparisons.
- Deliverable: `checkpoints/sense_final_standard_split.pth`.
- Verification: File exists, loads correctly via `load_checkpoint()`, and its logged validation accuracy is recorded in `data/processed/final_checkpoint_manifest.md`.

### Phase 7 Deliverables Checklist
- [ ] `configs/sense_full.yaml` finalized
- [ ] Full training run completed on standard split
- [ ] Light hyperparameter sweep documented if needed
- [ ] Final standard-split checkpoint selected and frozen

**Phase 7 exit criterion:** A trained, checkpointed SENSE model exists with documented validation performance, ready for full evaluation.

---

## Phase 8 — Evaluation Framework

### Objectives
- Build the metrics/reporting code once, generically, so it's reused identically across the standard-split result, every cross-scenario fold (Phase 9), and every ablation variant (Phase 10) — ensuring apples-to-apples comparison.

### Depends on
Phase 7 complete (needs at least one trained checkpoint to test the framework against).

### Tasks

**P8.1 — Implement core metrics module**
- Steps: `src/evaluation/metrics.py`:
  - `compute_overall_accuracy(preds, labels)`
  - `compute_per_class_accuracy(preds, labels, num_classes)` → returns dict + explicitly surfaces **minimum class accuracy** (this is the metric DARE reports at 92% — needed for direct comparison)
  - `compute_f1_scores(preds, labels, num_classes, average="macro")` and per-class F1
  - `compute_confusion_matrix(preds, labels, num_classes)` + a plotting helper (`matplotlib`/`seaborn` heatmap saved to file)
  - `measure_inference_time(model, dataloader, device, num_batches=50)` → returns mean ms/sample (needed for the "real-time viable" claim)
- Deliverable: `src/evaluation/metrics.py`.
- Verification: `tests/test_models.py` (or a new `tests/test_metrics.py`) — feed known synthetic predictions/labels with a hand-computed expected accuracy, assert the function matches.

**P8.2 — Implement standard evaluation script**
- Steps: `scripts/evaluate.py --checkpoint {path} --split {test_csv} --output {report_dir}`. Runs the model in eval mode over the specified split, computes all P8.1 metrics, saves a JSON results file + confusion matrix PNG + a human-readable Markdown summary.
- Deliverable: `scripts/evaluate.py`.
- Verification: Running on `checkpoints/sense_final_standard_split.pth` against the standard test split produces a complete report with all metrics populated (no `null`/`NaN` fields).

**P8.3 — Run Experiment 1 (standard evaluation) and record headline numbers**
- Steps: Execute P8.2 against `checkpoints/sense_final_standard_split.pth`. Save output to `results/experiment_1_standard/`.
- Deliverable: `results/experiment_1_standard/report.md`, `confusion_matrix.png`, `metrics.json`.
- Verification: Overall accuracy and min-class accuracy are recorded and directly comparable in format to DARE's reported 95.87% / 92%.

### Phase 8 Deliverables Checklist
- [ ] `src/evaluation/metrics.py` implemented and unit-tested
- [ ] `scripts/evaluate.py` implemented, generic across any checkpoint/split
- [ ] Experiment 1 (standard split) evaluation completed and recorded

**Phase 8 exit criterion:** `results/experiment_1_standard/` contains a complete, correct metrics report for the Phase 7 checkpoint, in a format directly comparable to DARE/VT-UHGR's published numbers.

---

## Phase 9 — Cross-Scenario Robustness Study (Experiment 2)

### Objectives
- Execute the thesis's primary novelty claim: systematic per-scenario robustness evaluation, something neither DARE nor VT-UHGR performed.

### Depends on
Phase 8 complete (evaluation framework ready), Phase 2.3 (scenario-holdout splits already built).

### Tasks

**P9.1 — Train 8 scenario-holdout models**
- Steps: For `i` in 1..8, train a fresh SENSE model on `scenario_holdout_{i}_train.csv` using the exact same `configs/sense_full.yaml` hyperparameters (only the data split differs — everything else held constant for fair comparison). This is **8 separate full training runs** — budget compute time accordingly; consider reducing epoch count slightly (e.g., 30 instead of 50) if compute time is a binding constraint, but document this deviation explicitly if made.
- Deliverable: `checkpoints/scenario_holdout_{1..8}_best.pth`.
- Verification: All 8 checkpoints exist and load correctly.

**P9.2 — Evaluate each holdout model on its held-out scenario**
- Steps: Run `scripts/evaluate.py` for each of the 8 checkpoints against its corresponding held-out test scenario. Collect into a single `src/evaluation/cross_scenario.py` aggregation script producing one combined table: scenario ID → accuracy, F1, min-class accuracy.
- Deliverable: `results/experiment_2_cross_scenario/per_scenario_results.csv`.
- Verification: All 8 rows populated; no missing scenarios.

**P9.3 — Analyze and visualize scenario difficulty ranking**
- Steps: Bar chart of accuracy by scenario, sorted descending. Cross-reference against scenario descriptions from `DATASET_SCHEMA.md` (P1.4) to identify *which physical conditions* (turbidity, low light, etc.) correlate with lower accuracy — this is the qualitative payoff of the whole experiment.
- Deliverable: `results/experiment_2_cross_scenario/scenario_difficulty_analysis.md` + bar chart figure.
- Verification: Analysis explicitly names which scenarios are hardest and offers a physically-grounded explanation (not just "scenario 4 is hardest" with no interpretation).

### Phase 9 Deliverables Checklist
- [ ] 8 scenario-holdout models trained
- [ ] Per-scenario evaluation results collected into one table
- [ ] Scenario difficulty analysis written with physical interpretation

**Phase 9 exit criterion:** A complete 8-row cross-scenario results table exists with written interpretation — this directly backs Claim 3 from the earlier fusion-justification discussion.

---

## Phase 10 — Ablation Studies (Experiments 3, 4, 5)

### Objectives
- Empirically justify every fusion decision in SENSE using the complementary-failure-mode framework established earlier — not just architectural description, but measured evidence.

### Depends on
Phase 7 complete (need the full-model training/eval pipeline proven), Phase 9 (reuses scenario-difficulty framing for Experiment 3's breakdown).

### Tasks

**P10.1 — Experiment 3: Enhancement ablation**
- Steps: Using `configs/sense_full.yaml` as base, create `configs/sense_no_enhancement.yaml` (`enhancement_method: none`) and reuse the `funie_gan` full run from Phase 7 as one arm; additionally train a `waternet` variant (`configs/sense_waternet.yaml`). Train the `none` variant fully (same schedule as Phase 7). Evaluate all three (`none`, `waternet`, `funie_gan`) on the standard test split **and** broken down per-scenario (reuse Phase 9's scenario test sets for evaluation only, no retraining needed here — just run each of these three checkpoints against each scenario's test subset).
- Deliverable: `results/experiment_3_enhancement/comparison_table.csv` (rows: none/waternet/funie_gan, columns: overall acc, per-scenario acc).
- Verification: Table populated; explicitly check whether the accuracy *gap* between `none` and `funie_gan` **widens** in harder scenarios (this is the specific claim from the earlier fusion-justification discussion — average benefit isn't enough, the widening-under-degradation pattern is what proves the mechanism).

**P10.2 — Experiment 4: Stereo vs mono ablation**
- Steps: Create `configs/sense_mono.yaml` (`stereo.enabled: false`). Train fully on standard split. Evaluate standard split **and** break down by individual gesture class (not just overall) — the earlier discussion specifically predicts stereo helps most on depth-ambiguous gesture pairs (e.g., gestures differing mainly in thumb-in-front vs thumb-behind configurations). Identify which CADDIAN classes those are from `DATASET_SCHEMA.md` and check if they show the largest stereo-vs-mono accuracy gap.
- Deliverable: `results/experiment_4_stereo/comparison_table.csv` + `results/experiment_4_stereo/per_class_gap_analysis.md`.
- Verification: Per-class breakdown explicitly identifies which gestures benefit most from stereo, and whether this matches the depth-ambiguity hypothesis.

**P10.3 — Experiment 5: Text guidance ablation**
- Steps: Create `configs/sense_no_text.yaml` (`text_guidance.enabled: false`, using the P5.7 fallback linear head). Train fully on standard split. Evaluate standard split **and** per-scenario (reuse Phase 9 scenario test sets). Specifically check whether the accuracy gap between `no_text` and full-text-guided SENSE **widens under harder scenarios** — same widening-under-degradation test as P10.1, since this is the specific mechanism claimed for text anchoring.
- Deliverable: `results/experiment_5_text_guidance/comparison_table.csv` + widening-gap analysis.
- Verification: Explicit numeric evidence for or against the "stabilizes under degradation" hypothesis — report honestly even if the hypothesis isn't fully supported; partial support or null results are legitimate and reportable findings, not failures.

**P10.4 — Consolidate all ablation results into one master table**
- Steps: `results/ablation_master_summary.md` — one table combining Experiments 3, 4, 5 overall-accuracy deltas, ready to drop into the thesis results chapter.
- Deliverable: `results/ablation_master_summary.md`.
- Verification: Every ablation arm has a recorded overall accuracy, min-class accuracy, and (where applicable) the widening-gap verdict.

### Phase 10 Deliverables Checklist
- [ ] Enhancement ablation (none/waternet/funie_gan) trained and evaluated
- [ ] Stereo vs mono ablation trained and evaluated with per-class breakdown
- [ ] Text guidance ablation trained and evaluated with widening-gap analysis
- [ ] Master ablation summary table produced

**Phase 10 exit criterion:** All three ablation experiments have trained models, evaluated results, and written interpretation directly addressing the complementary-failure-mode justifications established earlier in the project.

---

## Phase 11 — Comparative Analysis vs DARE & VT-UHGR

### Objectives
- Produce the final head-to-head comparison tables/figures that will anchor the thesis results chapter.

### Depends on
Phase 8, 9, 10 complete.

### Tasks

**P11.1 — Build the final comparison table**
- Steps: `results/final_comparison_table.md` — rows: Yang et al. baseline (from Phase 4), DARE (literature-reported 95.87%/92%), VT-UHGR (literature-reported, qualitative), SENSE (Phase 7 standard-split result). Columns: overall accuracy, min-class accuracy, enhancement used, stereo used, text guidance used, cross-scenario tested (yes/no).
- Deliverable: `results/final_comparison_table.md`.
- Verification: Every SENSE-side number traces back to a specific results file from Phases 4/8/9/10 (cite the exact JSON/CSV, don't hand-type numbers from memory).

**P11.2 — Build headline figures for the thesis**
- Steps: Generate: (a) bar chart comparing SENSE vs DARE vs VT-UHGR vs baseline on overall + min-class accuracy; (b) the per-scenario robustness chart from Phase 9; (c) the ablation summary chart from Phase 10.
- Deliverable: `results/figures/` — PNG/SVG exports at thesis-ready resolution (300dpi minimum).
- Verification: Figures render correctly, labels legible at print size, consistent color scheme across all charts.

**P11.3 — Write the honest limitations section**
- Steps: Document anywhere SENSE did **not** outperform DARE/VT-UHGR, or where an ablation didn't support its hypothesis. This is expected and normal for a rigorous thesis — document it plainly rather than omitting it.
- Deliverable: `results/limitations_and_honest_findings.md`.
- Verification: Every ablation/experiment from Phases 8–10 has an entry here, positive or negative.

### Phase 11 Deliverables Checklist
- [ ] Final comparison table built and fully traceable to source results
- [ ] Headline figures generated at thesis resolution
- [ ] Honest limitations section written

**Phase 11 exit criterion:** All results needed for the thesis results chapter exist as traceable files, figures are print-ready, and limitations are documented honestly.

---

## Phase 12 — Results Packaging & Thesis Writeup Support

### Objectives
- Convert raw results artifacts into report-ready material.

### Depends on
Phase 11 complete.

### Tasks

**P12.1 — Export all tables in report-ready format**
- Steps: Convert every `results/*.csv`/`.md` table into clean Markdown tables matching the formatting conventions already used in the thesis report sections built earlier in this project (background study, novelty statement, etc.) for direct copy-paste into the report document.
- Deliverable: `results/thesis_ready_tables.md`.
- Verification: Tables visually match the style of prior report sections (consistent header casing, column ordering).

**P12.2 — Draft the Results & Discussion section skeleton**
- Steps: Using the three publishable claims established earlier as the section's structuring spine, draft a Results section skeleton with placeholders pointing to the exact figures/tables from Phase 11, ready for the person to write prose around.
- Deliverable: `results/results_section_draft.md`.
- Verification: Skeleton references every figure/table produced; no orphaned or unused results.

**P12.3 — Archive all configs used for final reported numbers**
- Steps: Copy the exact YAML configs used to produce every headline number into `results/archived_configs/` (immutable snapshot, separate from the working `configs/` directory which may keep evolving).
- Deliverable: `results/archived_configs/`.
- Verification: Every number in `final_comparison_table.md` can be traced to an archived config + checkpoint pair.

### Phase 12 Deliverables Checklist
- [ ] Thesis-ready tables exported
- [ ] Results section skeleton drafted
- [ ] Final configs archived and traceable

**Phase 12 exit criterion:** A future reader (supervisor, examiner, or the student six months later) can reconstruct exactly how every reported number was produced.

---

## Phase 13 — Reproducibility & Handover

### Objectives
- Ensure the entire project can be reproduced from scratch by a third party, and is properly archived.

### Depends on
Phase 12 complete.

### Tasks

**P13.1 — Final README pass**
- Steps: Update `README.md` with: full project summary, how to reproduce every experiment (`scripts/train.py --config configs/{name}.yaml` for each named config), how to reproduce evaluation, link to `results/` directory structure explanation.
- Deliverable: Finalized `README.md`.
- Verification: A person unfamiliar with the project can reproduce Experiment 1's headline number by following README instructions alone.

**P13.2 — Freeze dependency versions**
- Steps: Run `pip freeze > requirements-lock.txt` from the final working environment, commit it alongside the looser `requirements.txt` for exact reproducibility.
- Deliverable: `requirements-lock.txt`.
- Verification: A clean venv installing from `requirements-lock.txt` reproduces the exact same package versions used for final results.

**P13.3 — Archive checkpoints and results externally**
- Steps: Since `checkpoints/` and `runs/` are git-ignored (correctly, given their size), export the final checkpoints (`sense_final_standard_split.pth`, 8 scenario-holdout checkpoints, 3 ablation checkpoints) to external storage (cloud drive / university storage) with a manifest file listing exactly what each checkpoint corresponds to.
- Deliverable: External archive + `checkpoints/ARCHIVE_MANIFEST.md` (committed to git, pointing to the external location).
- Verification: Manifest cross-references every checkpoint referenced in `results/final_comparison_table.md`.

**P13.4 — Final repository tag**
- Steps: `git tag -a v1.0-thesis-results -m "Final results for thesis submission"`.
- Deliverable: Git tag.
- Verification: `git show v1.0-thesis-results` shows the correct final commit.

### Phase 13 Deliverables Checklist
- [ ] README finalized and reproducibility-verified
- [ ] Dependency versions locked
- [ ] Checkpoints archived externally with manifest
- [ ] Final git tag created

**Phase 13 exit criterion:** Project is fully reproducible from a clean clone plus the external checkpoint archive, and is tagged as a stable final state.

---

## Appendix A — Config Flag Reference (for quick agent lookup)

| Flag | Location | Values | Effect |
|---|---|---|---|
| `data.enhancement_method` | dataset config | `none` \| `funie_gan` \| `waternet` | Which cached image variant `CaddyStereoDataset` loads |
| `model.stereo.enabled` | model config | `true` \| `false` | Whether `BiChannelEncoder` uses both branches (stereo) or left-only (mono) |
| `model.text_guidance.enabled` | model config | `true` \| `false` | Whether `SENSE` uses cross-modal fusion + cosine head, or fallback linear head |
| `model.shared_weights` | model config | `true` \| `false` | Whether the two `BiChannelEncoder` branches share weights |
| `training.loss.alpha` / `.beta` | training config | float | CE vs contrastive loss weighting |
| `training.phases` | training config | epoch ranges | 3-phase progressive unfreezing schedule boundaries |

## Appendix B — Task ID Quick Reference

Use the `P{phase}.{task}` IDs throughout this document to request specific work, e.g.:
- "Implement P0.13" → build the environment smoke test suite.
- "Implement P5.6" → assemble the full SENSE model class.
- "Implement P10.1" → run the enhancement ablation experiment.

## Appendix C — Non-Negotiable Checkpoints

These four gates must genuinely pass before proceeding past them — do not skip ahead even under time pressure, because failures here silently invalidate everything built on top:

1. **End of Phase 0:** CUDA + GPU verified working.
2. **End of Phase 1:** Dataset integrity report shows zero missing-file errors.
3. **End of Phase 4:** Baseline reproduction accuracy is plausible (not near-random) — proves the data pipeline itself is correct before any novel architecture is built on it.
4. **End of Phase 5:** All four `SENSE` mode combinations pass forward+backward pass tests within VRAM budget.
