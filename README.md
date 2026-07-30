# SENSE — Stereo-Enhanced kNowledge-guided Gesture Encoder

Underwater diver gesture recognition on the CADDY stereo-vision dataset.

**Python:** 3.10.11 | **Framework:** PyTorch 2.x + CUDA 12.1 | **Hardware:** RTX 4060 (8 GB VRAM)

---

## Project Overview

SENSE is a bi-channel EfficientNet-B4 stereo encoder pipeline for recognising 15 diver hand gestures (+ no-gesture) across 8 real-world underwater scenarios in the CADDY dataset. It combines:

1. **Underwater image enhancement** — FUnIE-GAN / WaterNet preprocessing.
2. **Stereo depth fusion** — independent left/right EfficientNet-B4 branches fused via self-attention.
3. **Semantic text guidance** — cross-modal attention between visual features and CADDIAN class text embeddings (sentence-BERT), followed by a cosine-similarity classification head.

Baselines targeted: Yang et al. (90.08% GoogleLeNet / 88.59% ResNet), DARE (95.87% overall / 92% min-class), VT-UHGR.

---

## Requirements

- Windows / Linux
- Python **3.10.x** (3.10.11 recommended)
- NVIDIA GPU with CUDA 12.x+ driver (RTX 4060 or equivalent, ≥ 8 GB VRAM)
- NVIDIA driver ≥ 525 (for CUDA 12.x compatibility)

---

## Setup (step-by-step)

Follow **exactly** in order. Every step has a verification command — do not skip.

### 1. Confirm Python 3.10.x

```bash
py -3.10 --version
# Expected: Python 3.10.x
```

If Python 3.10 is not installed, download and install from:
https://www.python.org/downloads/release/python-31011/

### 2. Create the virtual environment

```bash
py -3.10 -m venv .venv
```

**Activate (Windows PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**Activate (Windows CMD):**
```cmd
.venv\Scripts\activate.bat
```

**Activate (Linux/Mac):**
```bash
source .venv/bin/activate
```

**Upgrade pip immediately after activation:**
```bash
python -m pip install --upgrade pip setuptools wheel
```

**Verify:**
```bash
# Windows:
where python
# Linux/Mac:
which python
# Expected: path ending in .venv/Scripts/python (Windows) or .venv/bin/python (Linux/Mac)
```

### 3. Install PyTorch with CUDA 12.1 (RTX 4060)

> ⚠️ **IMPORTANT:** Install PyTorch *first*, before `requirements.txt`. This uses the CUDA wheel
> index to avoid accidentally installing the CPU-only build.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

This downloads ~2.5 GB — allow several minutes on a fast connection.

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

**Verify (no broken dependencies):**
```bash
pip check
```

### 5. Configure environment variables

```bash
# Copy the example file
cp .env.example .env
```

Edit `.env` with actual paths once you have the CADDY dataset downloaded (Phase 1). The
defaults work for development before data is available.

### 6. Verify GPU / CUDA setup

```bash
python scripts/check_env.py
```

**Expected output (example):**
```
Python: 3.10.11 (...)
Platform: Windows-10-...
PyTorch: 2.5.1+cu121
CUDA available: True
CUDA device: NVIDIA GeForce RTX 4060 Laptop GPU
CUDA capability: (8, 9)
VRAM (GB): 8.59
```

> 🚫 If `CUDA available: False`, **stop here** — do not proceed. Verify your NVIDIA driver
> version and that you used the `--index-url https://download.pytorch.org/whl/cu121` flag.

### 7. Install pre-commit hooks

```bash
pre-commit install
```

**Verify:**
```bash
pre-commit run --all-files
```

### 8. Run the environment smoke tests

```bash
pytest tests/test_env_smoke.py -v
```

**Expected:** 4 tests pass. The first run will download model weights for `all-MiniLM-L6-v2`
(~90 MB) and EfficientNet-B4 pretrained weights — internet access required once.

### 9. Download and verify the CADDY dataset

Download the CADDY dataset from the official release (see Phase 1 in `Final_plan.md`),
extract it into `data/raw/caddy/`, then update your `.env` with the correct paths.

Verify dataset integrity:

```bash
python scripts/verify_dataset.py
```

**Expected:** All checks pass with zero integrity errors.

### 10. Generate data manifests and splits

Split CSVs are **not** tracked in Git (they are git-ignored as generated artefacts).
After downloading the dataset, regenerate them with a single command:

```bash
python scripts/generate_splits.py
```

This generates the following files under `data/processed/`:

| File | Description |
|---|---|
| `manifest.csv` | Full canonical manifest (16,429 real stereo pairs) |
| `splits/standard_train.csv` | Standard split — 70% training set (11,499 rows) |
| `splits/standard_val.csv` | Standard split — 15% validation set (2,465 rows) |
| `splits/standard_test.csv` | Standard split — 15% test set (2,465 rows) |
| `splits/scenario_holdout_1_*.csv` | LOSO fold 1 — held-out: `biograd-A` |
| `splits/scenario_holdout_2_*.csv` | LOSO fold 2 — held-out: `biograd-B` |
| `splits/scenario_holdout_3_*.csv` | LOSO fold 3 — held-out: `biograd-C` |
| `splits/scenario_holdout_4_*.csv` | LOSO fold 4 — held-out: `brodarski-A` |
| `splits/scenario_holdout_5_*.csv` | LOSO fold 5 — held-out: `brodarski-B` |
| `splits/scenario_holdout_6_*.csv` | LOSO fold 6 — held-out: `brodarski-C` |
| `splits/scenario_holdout_7_*.csv` | LOSO fold 7 — held-out: `brodarski-D` |
| `splits/scenario_holdout_8_*.csv` | LOSO fold 8 — held-out: `genova-A` |

> **Reproducibility note:** The script uses `SEED=42` (set in `.env`). Re-running it on the
> same dataset always produces identical splits.

---

## Directory Structure

```
sense-project/
├── .venv/                          # git-ignored
├── .env                            # git-ignored (copy from .env.example)
├── .env.example                    # committed — template for .env
├── .gitignore
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml                  # black / ruff config
├── .pre-commit-config.yaml
├── README.md                       # this file
├── Final_plan.md                   # full implementation blueprint
├── data/
│   ├── raw/
│   │   └── caddy/                  # untouched original CADDY files (git-ignored)
│   └── processed/
│       ├── enhanced/               # cached FUnIE-GAN/WaterNet outputs (git-ignored)
│       └── splits/                 # train/val/test + scenario split manifests (git-ignored)
├── weights/
│   ├── funie_gan/                  # FUnIE-GAN pretrained generator weights
│   └── waternet/                   # WaterNet pretrained weights
├── src/
│   ├── config.py                   # centralized config loader (paths, env vars)
│   ├── data/
│   │   ├── caddy_dataset.py        # CaddyStereoDataset + get_dataloader()
│   │   ├── transforms.py           # train/eval augmentation pipelines (stereo-consistent)
│   │   └── splits.py               # manifest builder + split generators
│   ├── enhancement/
│   │   ├── funie_gan.py            # FUnIEGANEnhancer wrapper
│   │   └── waternet.py             # WaterNetEnhancer wrapper
│   ├── models/
│   │   ├── visual_encoder.py       # BiChannelEncoder + SingleImageClassifier (baseline)
│   │   ├── text_encoder.py         # GestureTextEncoder (sentence-BERT)
│   │   ├── attention.py            # SelfAttentionBlock
│   │   ├── fusion.py               # CrossModalFusion
│   │   ├── classifier_head.py      # CosineSimilarityHead
│   │   └── sense.py                # Full SENSE model (wires all components)
│   ├── training/
│   │   ├── losses.py               # SenseLoss (CE + InfoNCE contrastive)
│   │   ├── trainer.py              # Trainer + PhaseController (progressive unfreezing)
│   │   └── scheduler.py            # optimizer + scheduler factory
│   ├── evaluation/
│   │   ├── metrics.py              # accuracy, F1, confusion matrix, inference time
│   │   ├── cross_scenario.py       # scenario-holdout aggregation
│   │   └── ablations.py            # ablation result analysis helpers
│   └── utils/
│       ├── logging.py              # TensorBoard wrapper
│       ├── seed.py                 # set_seed() for full reproducibility
│       └── checkpoint.py           # save/load checkpoint utilities
├── scripts/
│   ├── check_env.py                # P0.6 GPU/CUDA verification
│   ├── verify_dataset.py           # P1.2 dataset integrity checker
│   ├── generate_splits.py          # P2.1-P2.3 regenerate all manifests & splits
│   ├── run_eda.py                  # P2.0 standalone EDA computation script
│   ├── cache_enhanced_images.py    # P3.5 GAN enhancement caching
│   ├── train.py                    # main training entry point (--config)
│   ├── evaluate.py                 # evaluation script (--checkpoint --split)
│   ├── run_ablation.py             # Phase 10 ablation runner
│   └── run_cross_scenario.py       # Phase 9 cross-scenario runner
├── configs/
│   ├── base.yaml                   # shared defaults
│   ├── baseline_transfer_learning.yaml
│   ├── sense_full.yaml             # primary SENSE experiment
│   ├── sense_no_enhancement.yaml   # ablation: no FUnIE-GAN
│   ├── sense_waternet.yaml         # ablation: WaterNet enhancement
│   ├── sense_mono.yaml             # ablation: mono mode
│   └── sense_no_text.yaml          # ablation: no text guidance
├── notebooks/
│   ├── README.md                    # describes notebook purposes (00-07)
│   └── (analysis notebooks created in later phases)
├── tests/
│   ├── test_env_smoke.py           # P0.13 environment smoke tests
│   ├── test_data.py                # Phase 2 data pipeline tests
│   ├── test_models.py              # Phase 5 model unit tests
│   └── test_training_step.py       # Phase 6 training infrastructure tests
├── runs/                           # TensorBoard logs (git-ignored)
└── checkpoints/                    # model weights (git-ignored)
```

---

## Running Tests

```bash
# Environment smoke tests (run after initial setup)
pytest tests/test_env_smoke.py -v

# All tests (run after completing relevant phases)
pytest tests/ -v
```

---

## Running Experiments

```bash
# Check GPU is available
python scripts/check_env.py

# Train (replace config with desired experiment)
python scripts/train.py --config configs/sense_full.yaml

# Resume from checkpoint
python scripts/train.py --config configs/sense_full.yaml --resume checkpoints/sense_full_last.pth

# Evaluate
python scripts/evaluate.py --checkpoint checkpoints/sense_final_standard_split.pth \
    --split data/processed/splits/standard_test.csv \
    --output results/experiment_1_standard/
```

---

## Common Issues

### `CUDA available: False` after installing PyTorch

- **Cause:** CPU-only wheel was installed (missing `--index-url` flag).
- **Fix:** Uninstall and reinstall with the correct index URL:
  ```bash
  pip uninstall torch torchvision -y
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  ```

### `CUDA available: False` — driver issue

- **Cause:** NVIDIA driver does not support CUDA 12.x.
- **Fix:** Update driver to ≥ 525. Check with `nvidia-smi` — look for "CUDA Version" ≥ 12.1 in the header.

### PyTorch version bounds in `requirements.txt`

- **Status:** `requirements.txt` specifies `torch>=2.2,<2.6` and `torchvision>=0.17,<0.21`, which is fully compatible with the `torch-2.5.1+cu121` wheel. Always install PyTorch with the `--index-url https://download.pytorch.org/whl/cu121` wheel index first before running `pip install -r requirements.txt`. Run `pip check` to confirm no version conflicts exist.

### `pre-commit run --all-files` fails on new files

- **Cause:** Skeleton files may have formatting or import ordering issues.
- **Fix:** Run `black .` then `ruff check . --fix` to auto-fix, then re-run pre-commit.

### Out of Memory (OOM) during training

- **Cause:** Batch size too large for 8 GB VRAM.
- **Fix:** Reduce `training.batch_size` in the YAML config (try 8 or 4). Alternatively, enable gradient checkpointing in Phase 6 (`trainer.py`).

---

## Phase Reference

| Phase | Description | Key Deliverable |
|-------|-------------|-----------------|
| 0 | Bootstrap & Environment Setup | Working CUDA environment |
| 1 | Data Acquisition & Auditing | CADDY dataset on disk |
| 2 | Data Pipeline, EDA & Preprocessing | `CaddyStereoDataset`, splits |
| 3 | Enhancement Integration | Cached FUnIE-GAN/WaterNet images |
| 4 | Baseline Reproduction | Yang et al. accuracy reproduced |
| 5 | SENSE Model Construction | Full model, all ablation modes |
| 6 | Training Infrastructure | Trainer, checkpointing, logging |
| 7 | Training & Hyperparameter Strategy | `sense_final_standard_split.pth` |
| 8 | Evaluation Framework | Metrics, confusion matrices |
| 9 | Cross-Scenario Study (LOSO) | 8-fold scenario robustness results |
| 10 | Ablation Studies | Enhancement / stereo / text ablations |
| 11 | Comparative Analysis | Final comparison vs DARE / VT-UHGR |
| 12 | Results Packaging | Thesis-ready tables and figures |
| 13 | Reproducibility & Handover | Frozen configs, tagged release |

See `Final_plan.md` for full task-level detail on every phase.
