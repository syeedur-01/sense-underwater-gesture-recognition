# plan.md
## SENSE — Stereo-Enhanced kNowledge-guided Gesture Encoder
### Research-Grade Implementation Blueprint for Underwater Diver Gesture Recognition on CADDY

---

### How to use this document

This is the single source of truth for building, training, evaluating, and publishing SENSE. Work through phases **in order**. Every task has an ID (`P{phase}.{task}`) so a future prompt can say **"Implement Phase 3 Task 2"** and the scope is unambiguous. Each task specifies **Objective, Depends on, Steps, Deliverable, Verification**. A task is not "done" until its Verification step actually passes — not until it "runs without crashing."

This revision strengthens the original engineering blueprint with the research methodology a thesis committee and a journal reviewer will both expect: explicit research questions and hypotheses, a defined evaluation/model-selection protocol, leakage-proof splitting (applied to P2.2/P2.3/P6.7/P7.2/P8.2/P9.1), a disciplined hyperparameter strategy, statistical validation across folds, full experiment tracking, and a complete inventory of publication artifacts. Phase 0 and Phase 1 are retained here in full, unabridged detail — identical in content to what has already been implemented and verified — so this document remains a single, complete, standalone reference rather than requiring cross-referencing an earlier draft. Nothing about the phase order, numbering, task content, or engineering philosophy has changed from what was already built or specified — this is strictly an enrichment pass from Phase 2 onward, not a rewrite.

**Status note:** Phases 0–5 are implemented and verified. Phase 0 (environment), Phase 1 (data audit), Phase 2 (EDA, splits, pipeline), Phase 3 (enhancement cache — FUnIE-GAN and WaterNet), Phase 4 (ResNet18 baseline; GoogLeNet was not reproduced — published number only), and Phase 5 (SENSE model components, assembly, all four ablation-mode switches, and the P5.8 VRAM probe — image size frozen at 224, batch size frozen at 16 per `configs/base.yaml`). Phase 6 is next.

---

## Revision Changelog (this pass)

This pass corrects the specific issues identified in the final implementation-readiness audit. No methodology, architecture, implementation order, model design, experimental protocol, coding style, task numbering, or deliverable format was changed. No new experiments were introduced. The following corrections were applied:

1. **WaterNet configuration.** Added the missing `configs/sense_waternet.yaml` to the P0.8 directory tree, P0.12's config-creation task (count updated 6 → 7), the Phase 0 deliverables checklist, and P10.1, which now references it explicitly.
2. **Notebook creation tasks.** Added explicit tasks that build the five previously-orphaned notebooks — P4.4 (`03_baseline_experiment_analysis.ipynb`), P7.5 (`05_hyperparameter_analysis.ipynb`), P8.4 (`06_final_evaluation.ipynb`, Experiment 1 portion), P9.5 (`07_robustness_analysis.ipynb`), P10.5 (`04_ablation_study.ipynb`) — plus P11.4, which finalizes `06_final_evaluation.ipynb` with the full comparison. Each phase's Deliverables Checklist now lists its notebook. The Notebooks table's "Produced in" column now cites the exact task ID for every notebook.
3. **Publishable Claims.** Added a "Publishable Claims" subsection (Claim 1–3) between Research Hypotheses and Evaluation Strategy, mapping directly to the existing RQs/hypotheses/experiments — resolving the previously undefined "three publishable claims" / "Claim 3" references.
4. **Dependency graph.** Phase 5 now explicitly depends on both Phase 3 and Phase 4 in the Phase Index table (matching Phase 4's own exit criterion, which already gated Phase 5), plus a one-line "Depends on" note added to the Phase 5 section itself.
5. **Ablation checkpoint count.** Corrected from "3 ablation variants" to the accurate 4 (`ablation_enhancement_none.pth`, `ablation_enhancement_waternet.pth`, `ablation_stereo_mono.pth`, `ablation_text_guidance_none.pth`) in Appendix D and P13.3, with explicit filenames now assigned in P10.1–P10.3 and consolidated in P10.4.
6. **LOSO methodology wording.** Added a Phase 10 methodology note, and revised H2, H4, P9.3 (point 3), P10.1, and P10.3 to state clearly that the enhancement and text-guidance ablations' "per-scenario" results come from a standard-split breakdown (grouping `standard_test.csv` by `scenario_id`), not from Phase 9's LOSO test sets — Phase 9 remains the project's only true LOSO experiment. No retraining under LOSO was added for any ablation.
7. **Test split wording.** Replaced ambiguous singular references to "the test split" with explicit standard-vs-LOSO wording in the Evaluation Strategy section, P6.8, the Phase 6 and Phase 8 exit criteria, P8.2, and Appendix C item 6.
8. **Text encoder configuration.** P5.3 now specifies that the text-embedding backbone is selected via `model.text_guidance.encoder_type` / `model.text_guidance.encoder_name` in YAML, never hardcoded; Appendix A gained both flags; P7.1 now states the project default (`sentence_bert` / `all-MiniLM-L6-v2`) explicitly for `configs/sense_full.yaml`. This is a configuration default, not a new research comparison.
9. **Directory tree.** P0.8's `notebooks/` entry now shows only `README.md` plus a placeholder comment, since individual notebooks are created by later phases, not Phase 0; `configs/` now includes `sense_waternet.yaml`.

---

## Project Snapshot

- **Goal:** Robust underwater diver gesture recognition on the CADDY stereo-vision dataset (16 gesture classes + no-gesture (17 classes total) = 17 classes total, 8 real-world scenarios).
- **Model:** SENSE — bi-channel EfficientNet-B4 stereo encoder → self-attention refinement → cross-modal attention fusion with CADDIAN class text embeddings → cosine-similarity classification head.
- **Baselines:** Yang et al. transfer learning (90.08% GoogleLeNet / 88.59% ResNet), DARE (95.87% overall / 92% min-class), VT-UHGR (best purely-visual comparator, exact % not published).
- **Novelty type:** Combinational/integrative — first pipeline combining (a) underwater image enhancement preprocessing, (b) explicit stereo depth fusion, (c) semantic text guidance, evaluated with (d) systematic cross-scenario robustness testing on CADDY.
- **Hardware:** RTX 4060, 8GB VRAM. Mixed precision required at chosen batch sizes.
- **Framework:** PyTorch.

---

## Research Questions

These are restated from the project's research phase and are the fixed reference point every experiment in this plan must trace back to.

- **RQ1 — Core Recognition Performance.** Which deep learning architecture, and specifically the proposed SENSE design, achieves the highest and most reliable accuracy in classifying CADDIAN gestures across CADDY's real-world underwater scenarios?
- **RQ2 — Enhancement Integration.** Does applying underwater image enhancement (FUnIE-GAN or WaterNet) before classification improve accuracy and robustness relative to unenhanced input, and does that improvement scale with scenario difficulty?
- **RQ3 — Stereo Depth Contribution.** Does fusing stereo depth information with RGB features improve gesture recognition performance over an RGB-only (monocular) baseline, and is the improvement concentrated in depth-ambiguous gesture classes?
- **RQ4 — Semantic Stability Under Degradation.** Does semantic text guidance stabilize visual representations specifically under degraded/harder scenarios, rather than providing only a uniform, scenario-independent gain? (Made explicit here, rather than left implicit, because Experiment 5 / Phase 10 is built specifically to answer it.)

## Research Hypotheses

Each hypothesis is directly testable by an experiment already defined later in this plan — no hypothesis is included unless a concrete phase/task produces the evidence to accept or reject it.

- **H1 (→ Experiment 2, Phase 9).** SENSE trained and evaluated with systematic Leave-One-Scenario-Out (LOSO) protocol will show measurably lower accuracy on scenarios exhibiting the most severe turbidity/low-light degradation than on the clearest scenarios, confirming that aggregate (non-scenario-stratified) accuracy reported by prior work (DARE, VT-UHGR) masks real robustness gaps.
- **H2 (→ Experiment 3, Phase 10).** Underwater image enhancement (FUnIE-GAN/WaterNet) preprocessing will produce a larger accuracy gain in harder scenarios than in easier ones — i.e., the benefit of enhancement is not uniform but widens under degradation. This is tested via a **standard-split, per-scenario breakdown** (see the Phase 10 methodology note), which compares module performance across scenarios using models trained on data from all 8 scenarios — it is **not** a true LOSO holdout result and is not interpreted as one. Rejection of this hypothesis (a flat, uniform benefit) is a legitimate and reportable outcome, not a failed experiment.
- **H3 (→ Experiment 4, Phase 10).** Stereo (bi-channel) fusion will improve accuracy disproportionately on gesture classes that are known to be visually ambiguous from a single 2D projection (e.g., gestures differing primarily by finger/thumb depth ordering), rather than uniformly across all 16 gesture classes.
- **H4 (→ Experiment 5, Phase 10).** Semantic text guidance will narrow the accuracy gap between SENSE and a visual-only ablation more under harder scenarios than under easier ones, consistent with text embeddings acting as a degradation-invariant anchor rather than a purely additive feature. As with H2, this is tested via a **standard-split, per-scenario breakdown**, not a true LOSO holdout result (see the Phase 10 methodology note).

---

## Publishable Claims

These are the three claims this project is designed to support with evidence, referenced elsewhere in this document (e.g. the Phase 9 exit criterion, P12.3). Each maps directly to RQs/hypotheses/experiments already defined above — no new research direction is introduced here, only a naming of what already exists.

- **Claim 1 — Core recognition performance.** SENSE's combinational architecture (enhancement + stereo fusion + semantic guidance) achieves the highest and most reliable recognition performance among compared methods on CADDY. Evidence: RQ1, Experiment 1 (Phase 8, standard-split result vs. Yang et al./DARE/VT-UHGR), reinforced by the ablation results (Experiments 3–5, Phase 10) showing each component's individual contribution, and by the backbone-matched control (P10.6) which separates backbone/recipe gains from contribution gains.
- **Claim 2 — Mechanism-specific module benefits.** Each of SENSE's three integrated components provides a targeted, mechanism-specific benefit rather than a generic accuracy bump: enhancement and semantic guidance disproportionately help under harder/degraded scenarios (RQ2/H2, RQ4/H4), while stereo fusion disproportionately helps on depth-ambiguous gesture classes (RQ3/H3). Evidence: Experiments 3, 4, and 5 (Phase 10).
- **Claim 3 — Cross-scenario robustness benchmark.** Systematic Leave-One-Scenario-Out evaluation reveals real-world robustness gaps across CADDY's 8 scenarios that aggregate, non-scenario-stratified accuracy reporting (as used by DARE and VT-UHGR) conceals. Evidence: RQ1/H1, Experiment 2 (Phase 9).

---

## Evaluation Strategy

This section is binding for every phase from Phase 6 onward. Any script or notebook that reports a "best" model, a "final" number, or a model-selection decision must follow this protocol exactly, or the number is not reportable in the thesis/paper.

**Primary metric — Macro F1-score.** Chosen over raw accuracy as the primary metric because CADDY's gesture classes are not perfectly balanced (confirmed during EDA, Phase 2) and macro F1 penalizes a model that performs well only on frequent classes while failing on rare ones — which is precisely the failure mode a safety-critical diver-AUV system cannot tolerate.

**Secondary metric — Overall (micro) accuracy.** Reported for direct comparability with DARE and VT-UHGR, both of which report overall accuracy as their headline number. Always reported alongside macro F1, never alone.

**Supporting metrics** (reported for every experiment, never used alone to select a checkpoint):
- Per-class Precision and Recall
- Per-class F1
- **Minimum class accuracy** — retained explicitly because DARE reports this (92%) and it is the more safety-relevant number than any averaged metric.
- Confusion matrix
- Inference latency (ms/sample) — supports the "real-time viable" claim, not used for model selection.

**Tie-breaking metric.** If two checkpoints/configurations are within 0.5 macro-F1 points of each other, the tie is broken by **minimum class accuracy** (favoring the model that fails least badly on its worst class), not by overall accuracy. This is a deliberate, documented choice — overall accuracy can be misleadingly won by a model that is simply better on the majority class.

**Best checkpoint selection criterion.** During training (Phase 6/7), the checkpoint saved as "best" at any epoch is the one with the **highest macro F1 on the validation split**, with minimum-class-accuracy as the tie-breaker per above. This selection **never** touches any test split — standard or LOSO (`standard_test.csv` or any `scenario_holdout_{i}_test.csv`) — in any phase, for any reason — see Phase 2/6/8 for the leakage-prevention mechanics enforcing this.

**Reporting convention.** Every results table in `results/` must report macro F1 and overall accuracy together, minimum class accuracy alongside them, and — wherever the result derives from more than one run (LOSO folds, repeated seeds) — mean ± standard deviation rather than a single point estimate (see Statistical Analysis section below).

---

## Phase Index

| Phase | Name | Depends On | Output |
|---|---|---|---|
| 0 | Bootstrap & Environment Setup | — | Reproducible dev environment |
| 1 | Data Acquisition & Auditing | 0 | Raw CADDY verified on disk |
| 2 | Data Pipeline, EDA & Preprocessing | 1 | Dataset understood, leakage-proof splits, `Dataset`/`DataLoader` |
| 3 | Enhancement Module Integration | 2 | FUnIE-GAN/WaterNet wrapper + cached enhanced dataset |
| 4 | Baseline Reproduction | 2 | Reproduced Yang et al. baseline (pipeline sanity check) |
| 5 | SENSE Model Construction | 3, 4 | Full model, shape- and gradient-tested |
| 6 | Training Infrastructure & Experiment Tracking | 5 | Trainer, loss, scheduler, checkpointing, full experiment logging |
| 7 | Phase-Wise Model Training & Hyperparameter Strategy | 6 | Trained SENSE checkpoint (standard split) |
| 8 | Evaluation Framework | 7 | Metrics suite bound to the Evaluation Strategy above |
| 9 | Cross-Scenario Robustness Study (LOSO) | 8 | Leave-one-scenario-out benchmark + statistical analysis |
| 10 | Ablation Studies | 7, 8, 9 | Enhancement / stereo-vs-mono / text-guidance ablations |
| 11 | Comparative Analysis vs DARE & VT-UHGR | 8, 9, 10 | Final comparison tables/figures |
| 12 | Publication Artifacts & Thesis Writeup Support | 11 | All figures/tables/notebooks inventoried and generated |
| 13 | Reproducibility & Handover | 12 | Final README, frozen configs, archived checkpoints, reproducibility checklist |

---

## Phase 0 — Project Bootstrap & Environment Setup

*(Already implemented and verified. Retained here in full detail — unchanged in content from the version executed — so the document remains a complete, standalone, auditable reference rather than requiring cross-referencing an earlier draft.)*

### Objectives
- Prepare a fully reproducible development environment.
- Ensure any contributor (human or AI agent) can run the project from scratch with zero ambiguity.

### Depends on
Nothing. This is the starting point.

### Tasks

**P0.1 — Select Python version**
- Objective: Lock a single interpreter version to avoid dependency drift.
- Steps:
  1. Use **Python 3.10.x** (best compatibility window across PyTorch 2.x, timm/EfficientNet weights, sentence-transformers, and CUDA 12.x wheels).
  2. Record the exact version in `.python-version` and in `README.md`.
- Deliverable: `.python-version` file.
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
  1. Use `requirements.txt`, pinning major.minor versions, not exact patch:
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
     scipy>=1.11
     ```
  2. `requirements-dev.txt` for tooling only:
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
  1. Check installed NVIDIA driver's supported CUDA version (`nvidia-smi`).
  2. Install PyTorch using the official CUDA 12.1 wheel index (RTX 4060 is Ada Lovelace, fully supported by CUDA 12.x):
     ```
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
     ```
  3. Do not install the CPU-only wheel by accident — verify in P0.6.
- Deliverable: PyTorch with CUDA installed inside `.venv`.
- Verification: See P0.6.

**P0.5 — Install supporting libraries**
- Objective: All non-DL-framework dependencies available.
- Steps:
  1. `pip install -r requirements.txt`
  2. `pip install -r requirements-dev.txt`
- Deliverable: Fully populated `.venv`.
- Verification: `pip list` shows all packages from both requirement files; `pip check` returns no broken requirements.

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
- Verification: Output shows `CUDA available: True`, device name containing "4060", VRAM ≈ 8.0 GB.

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
  3. Add `python-dotenv` loading utility in `src/config.py` (P0.9).
- Deliverable: `.env.example` (committed), `.env` (local only, git-ignored).
- Verification: A script importing `src/config.py` prints resolved paths without raising `KeyError`.

**P0.8 — Create initial project directory structure**
- Objective: A stable, unambiguous folder layout every later phase writes into.
- Steps: create exactly this structure (`.gitkeep` where needed to preserve empty dirs in git). Note: `notebooks/README.md` is created now and briefly summarizes the purpose of each planned notebook (00–07); the authoritative list is the Notebooks section later in this document. Individual notebook files are **not** created in Phase 0 — each is built by the specific phase task that owns it (see the "Produced in" column of the Notebooks section, e.g. P2.0, P4.4, P7.5, P8.4, P9.5, P10.5, P11.4).
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
  ├── pyproject.toml                  # black/ruff config only
  ├── data/
  │   ├── raw/
  │   │   └── caddy/                  # untouched original CADDY files
  │   └── processed/
  │       ├── enhanced/                # cached FUnIE-GAN/WaterNet outputs
  │       └── splits/                  # train/val/test + LOSO split manifests
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
  │   ├── sense_waternet.yaml
  │   ├── sense_mono.yaml
  │   ├── sense_no_text.yaml
  │   └── sense_neither.yaml
  ├── notebooks/
  │   ├── README.md                    # describes notebook purposes
  │   └── (analysis notebooks created in later phases)
  ├── tests/
  │   ├── __init__.py
  │   ├── test_data.py
  │   ├── test_models.py
  │   └── test_training_step.py
  ├── runs/                            # tensorboard logs, git-ignored
  └── checkpoints/                     # model weights, git-ignored
  ```
- Deliverable: Full directory tree committed (with `.gitkeep` in empty dirs), including `notebooks/README.md`.
- Verification: `find . -type d | sort` matches the structure above (minus git-ignored dirs' contents); `notebooks/` contains only `README.md` at this point — no `.ipynb` files yet, since those are created by later phases.

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
  2. Create `configs/base.yaml` with shared defaults (image size, num_classes=17, batch_size, etc.) that experiment-specific configs inherit/override.
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

**P0.11 — Add linting, formatting, pre-commit hooks**
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
- Verification: `pre-commit run --all-files` passes cleanly.

**P0.12 — Create configuration files (experiment configs)**
- Objective: Every experiment variant is defined declaratively — never as ad hoc code edits.
- Steps: create `configs/base.yaml` (P0.9) plus:
  - `configs/baseline_transfer_learning.yaml` — Phase 4 baseline.
  - `configs/sense_full.yaml` — full pipeline, Phase 7.
  - `configs/sense_no_enhancement.yaml`, `configs/sense_waternet.yaml`, `configs/sense_mono.yaml`, `configs/sense_no_text.yaml`, `configs/sense_neither.yaml` — Phase 10 ablations (stubs acceptable at this point, filled in later). `sense_no_enhancement.yaml` sets `data.enhancement_method: none`; `sense_waternet.yaml` sets `data.enhancement_method: waternet`; `sense_neither.yaml` disables stereo, text guidance, and enhancement (backbone-matched control, P10.6); the `funie_gan` arm is `sense_full.yaml` itself (no separate stub needed).
  - Each config specifies `model.*`, `data.*`, `training.*`, `enhancement.enabled`, `stereo.enabled`, `text_guidance.enabled` flags so ablations are config-switches, not code forks.
- Deliverable: All 8 YAML configs present.
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
- Verification: All 4 tests pass (note: `pretrained=True` and the sentence-transformer download require internet on first run — cache is local after that).

**P0.14 — Document setup instructions in README.md**
- Objective: Anyone (human or agent) can go from empty machine to verified environment by reading one file.
- Steps: Write `README.md` with sections: Project Overview, Requirements, Setup (step-by-step commands mirroring P0.1–P0.13), Directory Structure, Running Tests, Common Issues (e.g., CUDA mismatch).
- Deliverable: `README.md`.
- Verification: A fresh clone + following README top-to-bottom results in `pytest tests/test_env_smoke.py` passing with no undocumented manual steps.

### Phase 0 Deliverables Checklist
- [x] Working virtual environment (`.venv`, Python 3.10.x)
- [x] `requirements.txt` + `requirements-dev.txt` installed cleanly
- [x] `.env.example` (+ local `.env`)
- [x] Full project directory structure created
- [x] `src/config.py` functional
- [x] Git initialized, `.gitignore` correct, bootstrap commit made
- [x] Pre-commit hooks installed and passing
- [x] All 8 experiment config stubs present and parseable
- [x] `scripts/check_env.py` confirms CUDA + RTX 4060 detected
- [x] `tests/test_env_smoke.py` — all tests passing
- [x] `README.md` setup instructions verified by clean run-through

**Phase 0 exit criterion (met):** `pytest tests/` passes fully, `python scripts/check_env.py` shows CUDA + GPU detected, and a person unfamiliar with the project can set up the environment using only `README.md`.

---

## Phase 1 — Data Acquisition & Auditing

### Objectives
- Get the official CADDY dataset onto disk in a known-good state.
- Catalogue exactly what's inside it before writing any pipeline code against assumptions.

### Depends on
Phase 0 complete.

### Tasks

**P1.1 — Download official CADDY dataset**
- Steps: Download from the official CADDIAN site (not a Kaggle mirror — required for citation integrity): the complete dataset release (`CADDY_gestures_complete_v2_release.zip`), and, if separately obtained, the true-positives and true-negatives CSVs. Place under `data/raw/caddy/` exactly as downloaded, without renaming. Confirm all files are non-empty and of expected size (complete release ≈ 2.5GB; hands-ROI supplementary archive ≈ 1.5GB, if included).
- Deliverable: Raw files present in `data/raw/caddy/`.
- Verification: `ls data/raw/caddy/` shows the expected archive(s); file sizes are non-zero and within the expected range; if standalone CSVs were also downloaded separately, manually confirm (or have the agent confirm) they are identical in content/row-count to the CSVs bundled inside the complete release archive before treating either as canonical.

**P1.2 — Extract and inventory raw files**
- Steps: Extract `CADDY_gestures_complete_v2_release.zip` in place inside `data/raw/caddy/`, preserving whatever internal folder structure the archive actually contains (including any top-level nesting folder — do not assume a flat structure, confirm it). Extract any supplementary hands-ROI archive into a clearly separated subfolder (e.g., `data/raw/caddy/hands_rois/`), since it is a supplementary resource and not part of the core pipeline. Write `scripts/verify_dataset.py` that walks the extracted tree and reports: total image count, file extensions present, the real folder naming pattern (scenario folders, left/right subfolders, etc.), and the complete list of scenario folder names found.
- Deliverable: `scripts/verify_dataset.py`, console report.
- Verification: Reported image count is consistent with the paper's stated ~9,191 positive + ~7,190 negative stereo pairs (small discrepancies are acceptable across dataset releases; flag anything off by more than a few percent). Scenario folder count is exactly 8.

**P1.3 — Parse and validate CSV annotations**
- Steps: Identify the canonical CSVs — the ones bundled inside the extracted complete-release archive — and load them with Pandas inside `scripts/verify_dataset.py`. If standalone CSVs were also downloaded separately and confirmed identical (per P1.1), they may be discarded; do not maintain two copies as sources of truth. Validate: no missing gesture labels, exactly 16 gesture classes (IDs 0–15) in the positives file + no-gesture in the negatives file, every filename referenced in either CSV actually exists on disk at its real extracted path, and the scenario field has exactly 8 unique values.
- Deliverable: Validation report printed and saved to `data/processed/dataset_audit_report.txt`.
- Verification: Zero "missing file" errors; scenario count == 8; gesture class count == 16 (+1 negative class `true_neg`).

**P1.4 — Document dataset schema**
- Steps: Write `data/processed/DATASET_SCHEMA.md` describing: CSV column meanings, the actual folder layout found during extraction (including any nesting), the left/right image pairing convention (naming pattern linking a left image to its right counterpart), and a scenario ID → scenario description mapping (which of the 8 scenario folders correspond to which physical conditions — e.g., clear water, turbid, low light — extracted from the CADDY paper or release metadata, never guessed). List all 8 scenario folder names explicitly by their real names (e.g., `brodarski-D`, `genova-A`, etc.).
- Deliverable: `data/processed/DATASET_SCHEMA.md`.
- Verification: A teammate/agent with no prior CADDY exposure can locate any image's label and scenario using only this document.

### Phase 1 Deliverables Checklist
- [ ] Raw CADDY dataset downloaded and extracted from the official source
- [ ] Canonical CSV source resolved (bundled-in-archive vs. standalone duplicates reconciled, only one source of truth retained)
- [ ] `scripts/verify_dataset.py` runs clean with zero missing-file errors
- [ ] `data/processed/dataset_audit_report.txt` generated
- [ ] `data/processed/DATASET_SCHEMA.md` written and accurate, all 8 scenarios named explicitly
- [ ] `.env` updated to point at the real, extracted CSV/root paths (not placeholder filenames)

**Phase 1 exit criterion:** Running `python scripts/verify_dataset.py` produces a report with zero integrity errors, and `DATASET_SCHEMA.md` fully and accurately documents the real data layout.

---

## Phase 2 — Data Pipeline, Exploratory Data Analysis & Preprocessing

### Objectives
- Understand the dataset's real statistical properties **before** finalizing any split or pipeline decision (EDA deliberately sequenced ahead of split-generation, a documented deviation from a naive literal reading of "preprocessing," and the correct methodological order).
- Build leakage-proof three-way splits (train/val/test) for both the standard protocol and the Leave-One-Scenario-Out (LOSO) protocol.
- Build the `Dataset`/`DataLoader` pipeline consumed by every later phase.

### Depends on
Phase 1 complete.

### Tasks

**P2.0 — Exploratory Data Analysis (performed before split generation)**
- Objective: Surface class imbalance, scenario imbalance, and any data-quality issues that should inform splitting strategy, rather than discovering them after splits are already frozen.
- Steps:
  1. Build `notebooks/00_data_exploration.ipynb` (full contents specified in the Notebooks section below).
  2. Compute and record: per-gesture-class sample counts, per-scenario sample counts, image resolution statistics, and a manual visual spot-check of a sample of stereo pairs per scenario for corruption/misalignment.
  3. Write findings to `data/processed/eda_findings.md` — explicitly state whether class imbalance is present (and its severity), whether any scenario has a sample count too small to support a meaningful held-out test fold, and whether any corrupted/misaligned samples were found and excluded.
- Deliverable: `notebooks/00_data_exploration.ipynb` (executed, outputs saved), `data/processed/eda_findings.md`.
- Verification: Every downstream splitting decision (P2.1–P2.3) cites a specific finding from `eda_findings.md` — e.g., "stratification is applied because EDA found class counts ranging from X to Y."

**P2.1 — Build the canonical data manifest**
- Steps: `src/data/splits.py::build_manifest()` reads both canonical CSVs, produces one unified DataFrame: `sample_id, left_path, right_path, gesture_label, gesture_id, scenario_id, is_negative`. Save as `data/processed/manifest.csv`.
- Deliverable: `data/processed/manifest.csv`.
- Verification: `len(manifest)` matches P1.2's audited count; no nulls in required columns.

**P2.2 — Standard train/validation/test split (70/15/15, stratified)**
- Steps: `make_standard_split(manifest, val_size=0.15, test_size=0.15, seed=SEED)` using two chained stratified `train_test_split` calls (first carve off test, then split remainder into train/val), `stratify=manifest['gesture_id']` at each step. Save `standard_train.csv`, `standard_val.csv`, `standard_test.csv` under `data/processed/splits/`.
- Deliverable: Three split manifest CSVs.
- Verification: Per-class proportions differ by ≤2% across all three sets; zero `sample_id` overlap between any pair of the three sets (assert pairwise intersections are empty — this is the explicit anti-leakage check).

**P2.3 — Scenario-aware Leave-One-Scenario-Out (LOSO) splits with validation subset**
- Steps: `make_scenario_splits(manifest, val_fraction=0.15)` producing 8 folds. Per fold `i`: the held-out scenario is the **test** set in full, touched only in Phase 8/9 evaluation. The pooled data from the remaining 7 scenarios is stratified-split into `train`/`val` (val_fraction of the pooled 7-scenario subset). Save `scenario_holdout_{i}_train.csv`, `scenario_holdout_{i}_val.csv`, `scenario_holdout_{i}_test.csv` for `i` in 1..8.
- Deliverable: 24 CSV files (8 train/val/test triples).
- Verification: Per fold, `test` contains only the held-out scenario; `train` ∪ `val` contains zero samples from it; `train` ∩ `val` is empty. All three checks asserted programmatically, not eyeballed.

**P2.4 — `CaddyStereoDataset` (PyTorch `Dataset`)**
- Steps: Returns `{"left": tensor, "right": tensor, "label": int, "scenario_id": int, "sample_id": str}`. Supports an `enhanced_dir` argument (wired fully in Phase 3) to swap raw images for cached enhanced ones.
- Deliverable: `src/data/caddy_dataset.py`.
- Verification: `tests/test_data.py::test_dataset_getitem_shapes` — correct tensor shapes and valid label range.

**P2.5 — Stereo-consistent augmentation pipeline**
- Steps: Implement `train_transforms(image_size)` / `eval_transforms(image_size)` in Albumentations using `additional_targets={"image_right": "image"}`, so every transform in a single `Compose` call — spatial and photometric alike — samples one set of parameters and applies it identically to both stereo images; this is the mechanism enforcing the "identical on left/right" rule, not a manual per-side loop. `image_size` matches the EfficientNet-B4 backbone's expected input resolution (confirm via `timm`'s `default_cfg`; record the exact value in `configs/base.yaml`). Default `train_transforms()` pipeline, in order: `RandomResizedCrop(scale=(0.85, 1.0), ratio=(0.9, 1.1))` → `Rotate(limit=12, border_mode=BORDER_CONSTANT, p=0.5)` → `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5)` → `GaussianBlur(blur_limit=(3, 5), p=0.15)` → `GaussNoise(var_limit=(5.0, 20.0), p=0.15)` → `Normalize` (ImageNet mean/std, matching the pretrained backbone) → `ToTensorV2`. `eval_transforms()` is `Resize` → `Normalize` → `ToTensorV2` only — no randomness. Deliberately excluded, with this rationale recorded in a comment at the top of the file: **horizontal flip** (risks corrupting any direction-dependent CADDIAN gesture label and does not correctly preserve stereo geometry — check `DATASET_SCHEMA.md` before ever reconsidering); **CLAHE or any other contrast/enhancement-style operation** (would contaminate the `none` arm of Experiment 3/H2 — contrast/enhancement stays exclusively a Phase 3 concern, never folded into the on-the-fly augmentation pipeline); **synthetic underwater color-cast jitter and cutout/random erasing** (both considered and deferred — the former risks confounding the enhancement ablation the same way CLAHE would, the latter risks erasing the gesturing hand); **class-conditional augmentation intensity** (deliberately rejected in favor of one uniform policy for every class, to avoid two interacting class-frequency-dependent mechanisms — the actual imbalance mitigation lives in P2.9 instead).
- Deliverable: `src/data/transforms.py`.
- Verification: `tests/test_data.py::test_stereo_augmentation_consistency` — synthetic identical left/right pair, confirm identical transformation applied to both; a second assertion confirms `eval_transforms()` is fully deterministic (bit-identical output across repeated calls on the same input).

**P2.6 — `DataLoader` factory**
- Steps: `get_dataloader(split_csv, batch_size, shuffle, num_workers, transform)`, `pin_memory=True`.
- Deliverable: `get_dataloader()`.
- Verification: One batch iterates correctly with expected shapes, no exceptions.

**P2.7 — Preprocessing visualization notebook**
- Steps: `notebooks/01_preprocessing.ipynb` (full contents in Notebooks section below) — raw vs. augmented samples, image histograms.
- Deliverable: Executed notebook.
- Verification: Notebook runs end-to-end; before/after figures are visually correct (augmentation is actually happening, not silently a no-op).

**P2.8 — Split validation notebook**
- Steps: `notebooks/02_dataset_split_validation.ipynb` (full contents in Notebooks section below) — visual/statistical confirmation that P2.2 and P2.3's leakage checks hold, plus human-readable charts of class/scenario balance per split.
- Deliverable: Executed notebook, plus `data/processed/split_validation_report.md` summarizing pass/fail on every leakage check.
- Verification: Report explicitly states zero leakage detected across both the standard split and all 8 LOSO folds; any failed check blocks progression to Phase 3.

**P2.9 — Class-imbalance mitigation utility**
- Steps: `src/data/splits.py::compute_class_weights(train_csv_path, num_classes=17)` — computes inverse-frequency class weights from the given split's `gesture_id` distribution (`weight_c = N_total / (num_classes * count_c)`, normalized to mean 1.0), returned as a tensor indexed by class ID. This is a general-purpose utility, called fresh by every training run against whichever train split that run actually uses (Phase 7's standard split, each of Phase 9's 8 LOSO folds, each of Phase 10's ablation arms) — it is never a single frozen file reused across runs with different train-set class distributions. As a documented reference tied to the headline results, also compute and save `data/processed/class_weights_standard.md` — the weights for `standard_train.csv` specifically, with a one-line cross-reference to `eda_findings.md`'s imbalance finding.
- Deliverable: `compute_class_weights()` in `src/data/splits.py`, `data/processed/class_weights_standard.md`.
- Verification: `tests/test_data.py::test_class_weight_computation` — on a synthetic imbalanced CSV, the minority class receives the highest weight and all weights are strictly positive and finite; on a synthetic balanced CSV, weights are ≈uniform (all ≈1.0), confirming the mechanism is harmless when imbalance is absent.

### Phase 2 Deliverables Checklist
- [ ] EDA completed and documented before split generation
- [ ] `manifest.csv` built and validated
- [ ] Standard 70/15/15 stratified split created, leakage-checked
- [ ] 8 LOSO train/val/test triples created, leakage-checked
- [ ] `CaddyStereoDataset` implemented and unit-tested
- [ ] Stereo-consistent augmentation implemented per the fixed default pipeline and unit-tested, including determinism check
- [ ] `DataLoader` factory working
- [ ] Class-imbalance mitigation utility (`compute_class_weights()`) implemented and unit-tested, `class_weights_standard.md` produced
- [ ] Preprocessing and split-validation notebooks executed with saved outputs

**Phase 2 exit criterion:** `pytest tests/test_data.py` passes; `split_validation_report.md` shows zero leakage across every split; EDA findings are on record and explicitly referenced by the splitting decisions made.

---

## Phase 3 — Enhancement Module Integration

*(Unchanged in substance from the original blueprint; retained in full because it is load-bearing for RQ2/H2.)*

### Tasks
- P3.1 / P3.2 — Acquire and document pretrained FUnIE-GAN and WaterNet weights (source URLs recorded in `weights/*/SOURCE.md` — required for citability).
- P3.3 / P3.4 — `FUnIEGANEnhancer` / `WaterNetEnhancer` inference wrappers, shape- and range-tested.
- P3.5 — `scripts/cache_enhanced_images.py`: precompute and cache `none` / `funie_gan` / `waternet` variants for the **entire** dataset (never run GAN inference inside the training loop), written to `{ENHANCED_CACHE_DIR}/{method}/{sample_id}_{left|right}.png` — one subdirectory per method, filenames keyed to the same `sample_id` used in `manifest.csv`. After caching, the script runs two programmatic checks and writes `data/processed/enhancement_report.md`: (1) **completeness** — for every `sample_id` in `manifest.csv`, confirm a file exists in all three method subdirectories for both `left` and `right`; any missing file is listed by `sample_id` in the report, never silently dropped; (2) **stereo-pair integrity** — confirm the cached `left`/`right` filenames for every `sample_id` still correspond to that same `sample_id` after caching (guards against the two independent per-image processing passes — FUnIE-GAN/WaterNet are monocular — silently desynchronizing a pair). The report also records processing time/rate per variant.
- P3.6 — Visual QA notebook section comparing raw vs. both enhancement methods across all 8 scenarios (folded into `notebooks/01_preprocessing.ipynb`, see Notebooks section), extended with quantitative image statistics — mean brightness, contrast (pixel std), and a histogram — computed for a sample of images per scenario/method. Flag any enhanced image whose histogram std falls below 20% of its raw counterpart's as a possible degenerate GAN output, and list any flagged samples explicitly rather than relying on eyeballing the notebook alone.
- P3.7 — Wire `enhancement_method` config flag into `CaddyStereoDataset`.

### Phase 3 Deliverables Checklist
- [ ] Both enhancer wrappers implemented, tested, weights documented
- [ ] All three cached variants exist for the full dataset, verified complete against `manifest.csv` (not just spot-checked), documented in `enhancement_report.md`
- [ ] Stereo-pair integrity confirmed intact after caching for every sample
- [ ] Visual QA performed, backed by quantitative brightness/contrast/histogram statistics with a defined degenerate-output threshold
- [ ] Config-level switch verified

**Phase 3 exit criterion:** All three cached image sets exist for every sample in `manifest.csv` (verified programmatically, not just present as folders); `enhancement_report.md` confirms zero missing files and zero stereo-pair mismatches; dataset loads any variant via config; QA confirms sane, quantitatively-checked outputs with no unflagged degenerate cases.

---

## Phase 4 — Baseline Reproduction

### Objectives
Reproduce a known literature result (Yang et al. transfer learning) on the exact Phase 2 pipeline, as a pipeline-correctness gate before any novel architecture is built on top of it.

### Tasks
- P4.1 ► `SingleImageClassifier` (ResNet18 via `timm`, ImageNet-pretrained, 17-way head).
- P4.2 — Minimal training loop, `configs/baseline_transfer_learning.yaml` (`training.loss.class_weighted: false` — see P6.1's note on preserving literature comparability), left-image only, standard split's **train** for fitting and **val** for monitoring (test untouched).

  > **Checkpoint note (Fix 5):** The Phase 4 baseline trained on raw 224×224 left-only images (its own config); resolution differs from SENSE (decided in P5.8 — listed as a limitation). The checkpoint (epoch 17) was selected by **validation accuracy** (97.04 %), because the Phase 4 training loop logged accuracy only, not macro F1. This is a documented deviation from the Evaluation Strategy; it is acceptable because Phase 4 is a pipeline-correctness gate, not the headline result.
- P4.3 — Evaluate once on **test**, compare to literature (primarily Yang et al.'s ResNet figure 88.59%; GoogLeNet 90.08% is context only — published number, not reproduced), record in `data/processed/baseline_results.md`.
- P4.4 — Build and execute `notebooks/03_baseline_experiment_analysis.ipynb` (full contents in the Notebooks section below), consuming P4.2/P4.3's logged metrics and checkpoint: baseline training/validation loss & accuracy curves, checkpoint analysis, baseline performance summary table.

### Phase 4 Deliverables Checklist
- [ ] `SingleImageClassifier` implemented
- [ ] Baseline trained using train/val correctly, test touched only once at the end
- [ ] Baseline evaluation report written and compared to literature
- [ ] `notebooks/03_baseline_experiment_analysis.ipynb` executed with saved outputs

**Phase 4 exit criterion (non-negotiable gate):** Baseline test accuracy is plausible (not near-random). If it is near-random, the data pipeline has a bug that must be fixed before Phase 5 begins.

---

## Phase 5 — SENSE Model Construction

*(Unchanged in substance — build every component in isolation, shape/gradient-test it, then assemble.)*

**Depends on:** Phase 3 complete, and Phase 4's exit criterion met (baseline reproduction accuracy plausible — see Phase 4 exit criterion) — matches the Phase Index table above.

### Tasks
- P5.1 `BiChannelEncoder` (EfficientNet-B4 ×2, `shared_weights` flag from the start).
- P5.2 `SelfAttentionBlock` (single-token + residual + LayerNorm formulation, explicitly justified in a docstring — this is a design decision, not an incidental implementation detail, and must be defensible in the thesis).
- P5.3 `GestureTextEncoder` — implementation supports pluggable text-embedding backbones (Word2Vec / Sentence-BERT / BERT), selected entirely through `model.text_guidance.encoder_type` and `model.text_guidance.encoder_name` in the YAML config (see Appendix A) and never hardcoded in `src/models/text_encoder.py`; class names pulled from `DATASET_SCHEMA.md`, single source of truth; embeddings computed once and cached. `configs/sense_full.yaml` fixes the project default (`encoder_type: sentence_bert`, `encoder_name: all-MiniLM-L6-v2` — the same model already smoke-tested in P0.13), which is the configuration used for every headline SENSE result in this plan; this is a configuration default, not an open research variable, and Phase 10 does not introduce an encoder-comparison experiment.

  > **Fix 3 — Prompt construction:** Prompts come from `build_class_prompts(class_names, overrides)` (`src/models/text_encoder.py`): class names ordered by ID (0..16), underscores replaced by spaces, plus `model.text_guidance.class_prompt_overrides` (default: `true_neg: "no gesture"`). Raises `ValueError` if an override key is unknown, if any two prompts are identical, or if the prompt count ≠ 17. Verification: embeddings have shape (17, D); all rows distinct; `true_neg` row maps to "no gesture". Prompts are **fixed before SENSE training** and not tuned afterwards.
- P5.4 `CrossModalFusion` (bidirectional cross-attention; gradient flow tested).
- P5.5 `CosineSimilarityHead` (temperature scalar, fixed or learnable).
- P5.6 Full `SENSE` assembly, forward-tested on a real Phase 2/3 batch.
- P5.7 Ablation-mode switches (`stereo.enabled`, `text_guidance.enabled`) built as config-driven branches inside the **same** model class — required for Phase 10's ablations to be fair, apples-to-apples comparisons rather than separately-coded models that could silently diverge in implementation details.
- P5.8 Parameter count / VRAM budget check and image-size/batch-size selection (<7GB peak on RTX 4060). Runs on the fully assembled `SENSE` model (after P5.6 and P5.7).

  > **Image size and batch size decision:** The SENSE `data.image_size` and `training.batch_size` are chosen here by the VRAM check (EfficientNet-B4 native 380×380; two branches on 8 GB RTX 4060).
  >
  > **Selection rule:**
  > - Probe candidate image sizes 224, 320, 380 with mixed precision.
  > - Probe the heaviest mode (full SENSE: both EfficientNet-B4 branches, text guidance on) in training mode: forward + backward + optimizer step, using a throwaway loss and AdamW on synthetic data, since the Phase 6 trainer does not exist yet. Peak VRAM must stay under 7 GB.
  > - Minimum batch size is 16 (the contrastive loss needs reasonably sized batches).
  > - Choose the LARGEST candidate size that satisfies both constraints. If two sizes are close in VRAM, prefer the one with the shorter epoch time.
  >
  > **Outputs:** Write the probe table (size, batch size, peak VRAM, seconds per iteration) and the stated selection rule to `results/vram_probe.md`. Record the chosen values in `configs/base.yaml` or `configs/sense_full.yaml`.
  >
  > **Scope:** The chosen values are kept identical across all SENSE variants (full, no_enhancement, waternet, mono, no_text, neither), frozen before Phase 7, and not changed after seeing results. They feed `train_transforms(image_size)` / `eval_transforms(image_size)`. The probe selected 224×224, which matches the baseline's resolution — both run below EfficientNet-B4's native 380×380 due to the 8 GB VRAM budget; this is documented as a limitation in P11.3.

### Phase 5 Deliverables Checklist
- [x] All five core components implemented, unit-tested, gradients confirmed flowing
- [x] Full model assembled and forward-tested on real data
- [x] All four mode combinations (full / no-stereo / no-text / neither) tested
- [x] VRAM budget confirmed safe

**Phase 5 exit criterion:** `pytest tests/test_models.py` passes fully; all four ablation-mode combinations run forward+backward within VRAM budget on a real batch.

---

## Phase 6 — Training Infrastructure & Experiment Tracking

### Objectives
Build the reusable trainer, loss, scheduler, checkpointing, and logging systems once, with **full experiment reproducibility baked in from the start** — every later training run (Phases 4/7/9/10) reuses this exact infrastructure and produces an identically-structured, fully-traceable experiment record.

### Depends on
Phase 5 complete.
**Prerequisite:** P8.1 (`src/evaluation/metrics.py` and `tests/test_metrics.py`) is implemented before P6.4, because
checkpoint selection, the experiment tracker and the Trainer all need macro F1 and min-class accuracy every epoch.
All of Phase 6 imports these from `src/evaluation/metrics.py` and never defines its own metric code. P8.1 is then
already complete when Phase 8 starts.

### Tasks

**P6.1 — Combined loss function**
- Steps: `SenseLoss` = `alpha * CrossEntropy + beta * ContrastiveLoss` (InfoNCE-style image-text contrastive term). `CrossEntropy` accepts an optional `weight=` tensor from `compute_class_weights()` (P2.9), computed fresh from whichever `train_csv` the current run is using, applied when `training.loss.class_weighted: true` (default `true` for every SENSE run — Phase 7, each Phase 9 LOSO fold, each Phase 10 ablation arm — since the mechanism degrades gracefully to ≈uniform weighting on a balanced split, so leaving it on costs nothing). The Phase 4 baseline (`configs/baseline_transfer_learning.yaml`) explicitly sets `class_weighted: false` and keeps plain unweighted CE, to preserve direct comparability with the published baseline numbers it's benchmarked against (P4.3). Gracefully degrades to CE-only when `text_guidance.enabled=False`, independent of the class-weighting setting.
- Deliverable: `src/training/losses.py`.
- Verification: Positive, finite scalar; `.backward()` succeeds; on a synthetic imbalanced batch, weighted CE produces a numerically different loss value than unweighted CE, confirming the weight tensor is actually applied rather than silently ignored.

**P6.2 — Optimizer + scheduler factory**
- Steps: AdamW + `CosineAnnealingLR`, parameters from config.
- Deliverable: `src/training/scheduler.py`.
- Verification: LR trends downward over dummy iterations.

**P6.3 — Three-phase progressive unfreezing controller**
- Steps: Epoch-boundary-driven `requires_grad` toggling on `BiChannelEncoder` submodules, with optimizer parameter-group rebuild at each transition.
- Deliverable: `PhaseController` in `trainer.py`.
- Verification: `requires_grad` states correct at epochs 1, 11, 31 (per the default 10/20/20-epoch phase boundaries).

**P6.4 — Checkpointing utility**
- Steps: `save_checkpoint()` / `load_checkpoint()` persisting model, optimizer, scheduler state, epoch, **all metrics from the Evaluation Strategy section (macro F1, accuracy, min-class accuracy)**, and the exact YAML config used.
- Deliverable: `src/utils/checkpoint.py`.
- Verification: Bit-identical weights after a save/load round-trip.

**P6.5 — Full experiment tracking layer (new, research-grade addition)**
- Objective: Every training run — baseline, full SENSE, every LOSO fold, every ablation — produces a self-contained, reproducible experiment record without manual bookkeeping.
- Steps: Extend `src/utils/logging.py` with an `ExperimentTracker` that, for every run, automatically creates `runs/{experiment_name}/{timestamp}/` containing:
  - `config.yaml` — an exact copy of the config used for this run
  - `metrics.csv` — one row per epoch: train loss (total/CE/contrastive), val loss, val accuracy, val macro F1, val min-class accuracy, learning rate, epoch duration
  - TensorBoard event files (scalars as in the original plan, unchanged)
  - `training.log` — plain-text log of console output for this run
  - `checkpoints/best.pth` and `checkpoints/last.pth`
  - `seed.txt` — the exact seed used
  - `runtime.txt` — total wall-clock training time
  - `gpu_info.txt` — output of `nvidia-smi -q` captured at run start (GPU model, driver version, VRAM)
  - `git_commit.txt` — output of `git rev-parse HEAD` at run start (empty/"not available" if git is unreachable — never fail the run over this)
- Deliverable: `ExperimentTracker` class, wired into `Trainer`.
- Verification: `tests/test_training_step.py::test_experiment_tracker_outputs` — after a 1-epoch dummy run, all nine artifacts listed above exist and are non-empty in the run directory.

**P6.6 — TensorBoard logging**
- Steps: As original — train/val loss components, accuracy, macro F1, min-class accuracy, learning rate, epoch timing, fixed validation-batch qualitative tracking.
- Deliverable: `src/utils/logging.py` (TensorBoard portion).
- Verification: Populated scalar charts after a dummy run.

**P6.7 — Seed control**
- Steps: `set_seed(seed)` across Python/NumPy/PyTorch/CUDA, `cudnn.deterministic=True`.
- Deliverable: `src/utils/seed.py`.
- Verification: Identical epoch-1 loss across two runs with the same seed.

**P6.8 — Assemble the `Trainer` class**
- Steps: Ties together dataloaders, model, loss, optimizer/scheduler, phase controller, checkpointing, `ExperimentTracker`, early stopping. `.fit(num_epochs)` trains on **train** and evaluates on **val** at the end of every epoch for early stopping and checkpoint selection — **never** on any test split. `.validate()` is usable standalone against any split passed to it, but its docstring explicitly states it must only be called against `val` during training/tuning; each run's designated test split is reserved exclusively for its own evaluation phase — the standard test split for Phase 8, and each fold's held-out scenario test split for Phase 9.
- Deliverable: `src/training/trainer.py`.
- Verification: `tests/test_training_step.py::test_trainer_single_epoch` — one epoch on a tiny subset completes, logs correctly, saves a checkpoint, loss finite.

**P6.9 — `scripts/train.py` CLI**
- Steps: `--config configs/{name}.yaml [--resume ...] [--experiment-name ...]`.
- Deliverable: `scripts/train.py`.
- Verification: 1-epoch smoke test on `configs/sense_full.yaml` completes end-to-end and produces a full experiment-tracking directory per P6.5.

### Phase 6 Deliverables Checklist
- [ ] Combined loss implemented and tested
- [ ] Optimizer/scheduler factory implemented
- [ ] Progressive unfreezing controller implemented and tested
- [ ] Checkpointing round-trip verified
- [ ] Full experiment tracking (`ExperimentTracker`) implemented and verified producing all nine artifacts
- [ ] TensorBoard logging verified
- [ ] Seed reproducibility verified
- [ ] `Trainer` assembled, correctly separates train/val/test usage, one-epoch smoke test passes
- [ ] `scripts/train.py` functional end-to-end

**Phase 6 exit criterion:** A 1-epoch smoke test of `scripts/train.py --config configs/sense_full.yaml` produces a complete, self-contained experiment record (all nine P6.5 artifacts present) and never accesses any test split.

---

## Phase 7 — Phase-Wise Model Training & Hyperparameter Strategy

### Objectives
Train SENSE to convergence on the standard split, using a disciplined, thesis-appropriate (not exhaustive) hyperparameter strategy, producing the primary checkpoint for all headline comparisons.

### Depends on
Phase 6 complete.

### Tasks

**P7.1 — Baseline configuration**
- Steps: Finalize `configs/sense_full.yaml` with a single, justified starting configuration: batch size (from P5.8's VRAM budget), 50-epoch 3-phase schedule, `alpha=0.7 / beta=0.3` loss weights, `enhancement_method: funie_gan`, `stereo.enabled: true`, `text_guidance.enabled: true`, `text_guidance.encoder_type: sentence_bert`, `text_guidance.encoder_name: all-MiniLM-L6-v2`, `training.loss.class_weighted: true`, standard split paths. This is the **baseline configuration** referred to in all later hyperparameter documentation.
- Deliverable: Finalized `configs/sense_full.yaml`.
- Verification: Config loads; every referenced path exists.

**P7.2 — Run full training**
- Steps: `python scripts/train.py --config configs/sense_full.yaml`. Confirm before starting that the val-split (not test-split) is wired into the epoch-end evaluation call — check the dataloader construction in the run config, not just assume it. Monitor via TensorBoard.
- Deliverable: A complete experiment run under `runs/sense_full/{timestamp}/` (per P6.5), including `checkpoints/best.pth`.
- Verification: Training completes all 50 epochs or triggers early stopping sensibly (not prematurely); validation macro F1 trend is visibly improving in TensorBoard/`metrics.csv`.

**P7.3 — Targeted sanity sweep (not exhaustive search)**
- Objective: If P7.2's validation performance clearly underperforms expectation, run a small, deliberate, fully-documented set of alternative configurations — never a grid search.
- Steps: If needed, run **at most 3–4** targeted variations, each changing exactly one axis at a time so the effect is attributable:
  1. Loss weighting: e.g., `alpha=0.5/beta=0.5` vs. the baseline `0.7/0.3`.
  2. Base learning rate: one alternative value (e.g., half or double the baseline).
  3. Batch size: one alternative constrained by the P5.8 VRAM budget.
  4. (Only if still needed) unfreezing schedule boundaries (e.g., 15/20/15 instead of 10/20/20 epochs).
  For every variation tried, explicitly record in `data/processed/hparam_sweep_notes.md`:
  - **What was changed** (the single axis, with old and new value)
  - **Why it was changed** (the specific symptom from P7.2's run that motivated it — e.g., "validation loss plateaued after epoch 8 with no accuracy gain, suggesting LR too high")
  - **What metrics were recorded** (val macro F1, val accuracy, val min-class accuracy, train/val loss curves)
  - **How the winning configuration is selected** (per the Evaluation Strategy section: highest val macro F1, min-class-accuracy as tie-breaker)
  - **What evidence is generated** (a comparison plot of val macro F1 vs. epoch across all tried variations, saved to `results/hparam_sweep/comparison_plot.png`, plus a summary table)
- Deliverable: `data/processed/hparam_sweep_notes.md`, `results/hparam_sweep/comparison_plot.png`, `results/hparam_sweep/summary_table.csv`.
- Verification: Every variation has a distinct experiment-tracking directory (per P6.5) and a written justification; the final chosen configuration is explicitly named and justified in writing, never selected by "whichever ran last."

**P7.4 — Select and freeze the final standard-split checkpoint**
- Steps: Based on P7.2 (or P7.3 if a sweep was needed), select the checkpoint with the best validation macro F1 (min-class accuracy as tie-breaker, per the Evaluation Strategy). Copy to `checkpoints/sense_final_standard_split.pth`.
- Deliverable: `checkpoints/sense_final_standard_split.pth`, an entry in `data/processed/final_checkpoint_manifest.md` recording its validation metrics and the exact experiment-run directory it came from.
- Verification: File loads correctly; manifest entry is fully traceable to a specific `runs/.../` directory.

**P7.5 — Hyperparameter analysis notebook**
- Objective: Turn P7.3's sweep records into a reviewable, visual comparison.
- Steps: Build and execute `notebooks/05_hyperparameter_analysis.ipynb` (full contents in the Notebooks section below), consuming `data/processed/hparam_sweep_notes.md`, `results/hparam_sweep/comparison_plot.png`, and `results/hparam_sweep/summary_table.csv` produced in P7.3 — learning-rate/batch-size/loss-weight comparison plots, early-stopping behavior analysis, best-configuration summary table. If P7.3 was skipped because P7.2 met expectations on the first run, the notebook instead documents the single baseline configuration and states explicitly that no sweep was needed.
- Deliverable: Executed `notebooks/05_hyperparameter_analysis.ipynb`.
- Verification: Notebook runs end-to-end from already-generated P7.3 outputs (or documents their absence per the note above); no training/evaluation logic duplicated inside the notebook.

### Phase 7 Deliverables Checklist
- [ ] Baseline configuration finalized and justified
- [ ] Full training run completed on standard split, val-only monitoring confirmed
- [ ] Sanity sweep (if performed) documented per the five required fields for every variation
- [ ] Final checkpoint selected per the Evaluation Strategy's stated criterion and frozen
- [ ] `notebooks/05_hyperparameter_analysis.ipynb` executed with saved outputs

**Phase 7 exit criterion:** A trained, checkpointed SENSE model exists with fully documented, traceable validation performance and a written justification for every configuration decision made.

---

## Phase 8 — Evaluation Framework

### Objectives
Build the metrics/reporting code once, generically, bound explicitly to the Evaluation Strategy defined above, and reused identically across the standard-split result, every LOSO fold, and every ablation.

### Depends on
Phase 7 complete.

### Tasks

**P8.1 — Core metrics module**
 (Implemented early as a Phase 6 prerequisite.)
- Steps: `src/evaluation/metrics.py`:
  - `compute_overall_accuracy`
  - `compute_macro_f1` (primary metric)
  - `compute_per_class_accuracy` (surfaces minimum class accuracy explicitly)
  - `compute_precision_recall_f1_per_class`
  - `compute_confusion_matrix` + heatmap plotting helper
  - `measure_inference_time(model, dataloader, device, num_batches=50)` → mean ms/sample
- Deliverable: `src/evaluation/metrics.py`.
- Verification: `tests/test_metrics.py` — hand-computed synthetic examples match function outputs exactly.

**P8.2 — Standard evaluation script**
- Steps: `scripts/evaluate.py --checkpoint {path} --split {test_csv} --output {report_dir}`. Runs eval mode over the specified split, computes all P8.1 metrics, saves JSON + confusion matrix PNG + Markdown summary.
- Note: `{test_csv}` here must always resolve to a designated test split — `standard_test.csv`, the relevant `scenario_holdout_{i}_test.csv`, or (per the Phase 10 methodology note) a scenario-grouped subset of `standard_test.csv` — never a `_val.csv` file. This same script is reused in Phase 9 (LOSO) and Phase 10 (ablations); Phase 8 is the **first** point in the project at which any test split is touched.
- Deliverable: `scripts/evaluate.py`.
- Verification: Running on `checkpoints/sense_final_standard_split.pth` against `standard_test.csv` produces a complete report, all metrics populated, no `null`/`NaN` fields.

**P8.3 — Run Experiment 1 (standard evaluation) and record headline numbers**
- Steps: Execute P8.2 against `checkpoints/sense_final_standard_split.pth`. Save to `results/experiment_1_standard/`.
- Deliverable: `results/experiment_1_standard/report.md`, `confusion_matrix.png`, `metrics.json`.
- Verification: Overall accuracy, macro F1, and min-class accuracy are all recorded, directly comparable in format to DARE/VT-UHGR's published numbers.

**P8.4 — Final evaluation notebook (Experiment 1 portion)**
- Objective: Give the standard-split evaluation visuals a concrete owner; this notebook is extended again in Phase 11 (P11.4) once the full comparison exists.
- Steps: Build and execute `notebooks/06_final_evaluation.ipynb` (full contents in the Notebooks section below), consuming `results/experiment_1_standard/metrics.json` and `confusion_matrix.png` from P8.3 — confusion matrix, classification report, precision/recall/macro-F1, per-class performance.
- Deliverable: Executed `notebooks/06_final_evaluation.ipynb` (Experiment 1 sections populated).
- Verification: Notebook runs end-to-end from P8.3's already-generated outputs; no evaluation logic duplicated inside the notebook.

### Phase 8 Deliverables Checklist
- [x] `metrics.py` implemented and unit-tested against hand-computed examples
- [ ] `evaluate.py` generic across any checkpoint/split, test-split-only enforced by convention/note
- [ ] Experiment 1 evaluation completed and recorded
- [ ] `notebooks/06_final_evaluation.ipynb` created and executed (Experiment 1 portion)

**Phase 8 exit criterion:** `results/experiment_1_standard/` contains a complete, correct metrics report, directly comparable to published baselines, and this is the first point in the project at which a test split — specifically, the standard test split — has been accessed.

---

## Phase 9 — Cross-Scenario Robustness Study (Experiment 2) & Statistical Analysis

### Objectives
Execute the thesis's primary novelty claim — systematic Leave-One-Scenario-Out (LOSO) robustness evaluation — and back it with proper statistical treatment across folds, testing **H1** directly.

### Depends on
Phase 8 complete; Phase 2.3 (LOSO splits).

### Tasks

**P9.1 — Train 8 scenario-holdout models**
- Steps: For `i` in 1..8, train a fresh SENSE model using `scenario_holdout_{i}_train.csv` for training and `scenario_holdout_{i}_val.csv` for epoch-wise validation/early stopping, using the exact same `configs/sense_full.yaml` hyperparameters (only the data split differs — everything else held constant for fair comparison). This is **8 separate full training runs** — budget compute time accordingly; consider reducing epoch count slightly (e.g., 30 instead of 50) if compute time is a binding constraint, but document this deviation explicitly if made.
- Deliverable: `checkpoints/scenario_holdout_{1..8}_best.pth`, each with its own full P6.5 experiment-tracking directory.
- Verification: All 8 checkpoints exist and load correctly; all 8 have complete, traceable experiment records.

**P9.2 — Evaluate each holdout model on its held-out scenario**
- Steps: Run `scripts/evaluate.py` for each checkpoint against its corresponding held-out `scenario_holdout_{i}_test.csv`. Aggregate via `src/evaluation/cross_scenario.py` into one table: scenario ID → accuracy, macro F1, min-class accuracy.
- Deliverable: `results/experiment_2_cross_scenario/per_scenario_results.csv`.
- Verification: All 8 rows populated.

**P9.3 — Statistical analysis across LOSO folds (new, research-grade addition)**
- Objective: Report LOSO results as a statistically defensible summary, not eight disconnected numbers.
- Steps:
  1. Compute **mean ± standard deviation** across the 8 folds for overall accuracy, macro F1, and min-class accuracy.
  2. Compute a **95% confidence interval** for the mean accuracy and mean macro F1 across folds (using the t-distribution given the small fold count, `scipy.stats.t.interval`).
  3. For Phase 10's standard-split, per-scenario comparisons (SENSE-full vs. each ablation arm — both trained on the standard split and evaluated on scenario-grouped subsets of `standard_test.csv`, per the Phase 10 methodology note; this is a different protocol from the LOSO folds described in this task and must not be presented as a LOSO comparison), run a **paired statistical test** (paired t-test or Wilcoxon signed-rank test, whichever is appropriate given the sample size and normality — check with a Shapiro-Wilk test first and justify the choice in writing) across the 8 scenario groups, rather than comparing only mean values.
  4. Document exactly when each of these analyses is performed: mean±SD and CI (for Phase 9's true LOSO folds) are computed immediately after P9.2; the paired test over Phase 10's 8 scenario groups is computed once the corresponding ablation variant's standard-split, per-scenario results exist (cross-referenced in Phase 10).
- Deliverable: `results/experiment_2_cross_scenario/statistical_summary.md` containing mean±SD, CI, and (once available) paired-test results and p-values with the test choice justified.
- Verification: Every reported LOSO number in the thesis traces back to this file, never to a single fold's result presented in isolation.

**P9.4 — Analyze and visualize scenario difficulty ranking**
- Steps: Bar chart of accuracy by scenario, sorted descending, with error bars if repeated-seed variance is available. Cross-reference `DATASET_SCHEMA.md` scenario descriptions to identify which physical conditions (turbidity, low light, etc.) correlate with lower accuracy.
- Deliverable: `results/experiment_2_cross_scenario/scenario_difficulty_analysis.md` + figure.
- Verification: Explicitly names the hardest scenarios and interprets them physically — this is the evidentiary basis for accepting or rejecting **H1**.

**P9.5 — Robustness analysis notebook**
- Objective: Give the LOSO robustness visuals a concrete owner.
- Steps: Build and execute `notebooks/07_robustness_analysis.ipynb` (full contents in the Notebooks section below), consuming `results/experiment_2_cross_scenario/per_scenario_results.csv`, `statistical_summary.md`, and `scenario_difficulty_analysis.md` from P9.2–P9.4 — scenario-wise and fold-wise performance, mean±SD/CI visualization, robustness discussion, failure-case analysis.
- Deliverable: Executed `notebooks/07_robustness_analysis.ipynb`.
- Verification: Notebook runs end-to-end from already-generated P9.2–P9.4 outputs; no LOSO training/evaluation logic duplicated inside the notebook.

### Phase 9 Deliverables Checklist
- [ ] 8 LOSO models trained, each fully tracked
- [ ] Per-scenario results aggregated
- [ ] Statistical summary (mean±SD, CI) computed and documented
- [ ] Scenario difficulty analysis written with physical interpretation, explicit H1 verdict
- [ ] `notebooks/07_robustness_analysis.ipynb` executed with saved outputs

**Phase 9 exit criterion:** A complete, statistically-summarized 8-fold LOSO result exists with an explicit accept/reject verdict on H1 — this is the evidentiary backbone for Claim 3 (cross-scenario benchmark).

---

## Phase 10 — Ablation Studies (Experiments 3, 4, 5)

### Objectives
Empirically test **H2, H3, H4** using the complementary-failure-mode framework established during the fusion-justification discussion — measured evidence, not architectural description alone.

**Methodology note (standard-split evaluation, not LOSO):** Experiments 3 and 5 (P10.1, P10.3) train and evaluate their ablation arms on the **standard** train/validation/test split, exactly as Phase 7 does — no ablation arm is retrained under the Leave-One-Scenario-Out protocol. Where a "per-scenario" breakdown is reported for these arms, it is produced by grouping `standard_test.csv` samples by `scenario_id`, **not** by reusing Phase 9's `scenario_holdout_{i}_test.csv` files. Because these models were trained on data drawn from all 8 scenarios, this breakdown compares module performance across scenarios — it is **not** a true held-out-scenario generalization result and must never be reported or interpreted as LOSO robustness. Phase 9 remains the project's only true LOSO experiment.

### Depends on
Phase 7 (supplies the `sense_full` checkpoint reused as the `funie_gan` arm), Phase 8 (supplies the `evaluate.py`/`metrics.py` evaluation framework this phase reuses), and Phase 9 (supplies the paired-test/Shapiro-Wilk statistical methodology this phase's standard-split, per-scenario comparisons also apply — per the Phase 10 methodology note below, this phase does **not** reuse Phase 9's scenario-holdout test sets or trained checkpoints).

### Tasks

**P10.1 — Experiment 3: Enhancement ablation (tests H2)**
- Steps: Train `configs/sense_no_enhancement.yaml` (`none`) and `configs/sense_waternet.yaml` (`waternet`) variants fully on the standard split, same schedule as Phase 7 (`funie_gan` arm already exists from Phase 7 as `checkpoints/sense_final_standard_split.pth`). Evaluate all three on the standard test split overall **and** on a per-scenario breakdown obtained by grouping `standard_test.csv` samples by `scenario_id` (per the Phase 10 methodology note above — this is a standard-split comparison, not a LOSO evaluation, and Phase 9's `scenario_holdout_{i}_test.csv` files are not reused here).
- Deliverable: `checkpoints/ablation_enhancement_none.pth`, `checkpoints/ablation_enhancement_waternet.pth`, `results/experiment_3_enhancement/comparison_table.csv`.
- Verification: Explicitly test H2 — check whether the accuracy **gap** between `none` and `funie_gan` widens on the harder-scenario subsets of the standard test split (per the statistical method in P9.3, applied here comparing `none` vs. `funie_gan` per scenario-grouped subset). Report the paired-test result and accept/reject H2 explicitly, not just "enhancement helps." State explicitly in the report that this reflects standard-split, per-scenario performance, not LOSO robustness.

**P10.2 — Experiment 4: Stereo vs. mono ablation (tests H3)**
- Steps: Train `configs/sense_mono.yaml` (`stereo.enabled: false`) fully. Evaluate standard split **and** per-gesture-class breakdown. Cross-reference `DATASET_SCHEMA.md` to identify which CADDIAN classes are depth-ambiguous (differ mainly by thumb/finger depth ordering) and check whether those specific classes show the largest stereo-vs-mono gap.
- Deliverable: `checkpoints/ablation_stereo_mono.pth`, `results/experiment_4_stereo/comparison_table.csv` + `per_class_gap_analysis.md`.
- Verification: Explicit accept/reject verdict on H3, with the specific gesture classes named.

**P10.3 — Experiment 5: Text guidance ablation (tests H4)**
- Steps: Train `configs/sense_no_text.yaml` (`text_guidance.enabled: false`, P5.7 fallback linear head) fully on the standard split. Evaluate standard split overall **and** on a per-scenario breakdown obtained by grouping `standard_test.csv` samples by `scenario_id` (per the Phase 10 methodology note above — Phase 9's `scenario_holdout_{i}_test.csv` files are not reused here). Apply the same widening-gap statistical test as P10.1.
- Deliverable: `checkpoints/ablation_text_guidance_none.pth`, `results/experiment_5_text_guidance/comparison_table.csv` + widening-gap analysis.
- Verification: Explicit accept/reject verdict on H4. A null or partial result is a legitimate, reportable finding — report it honestly rather than reframing it as support it doesn't have. State explicitly in the report that this reflects standard-split, per-scenario performance, not LOSO robustness.

**P10.4 — Consolidate all ablation results**
- Steps: `results/ablation_master_summary.md` combining Experiments 3–5 and P10.6's overall accuracy, macro F1, min-class accuracy deltas, hypothesis verdicts (H2, H3, H4), and the backbone-matched control's Claim 1 attribution, ready for the thesis results chapter. Explicitly list all **five** new ablation checkpoints trained in this phase — `checkpoints/ablation_enhancement_none.pth`, `checkpoints/ablation_enhancement_waternet.pth`, `checkpoints/ablation_stereo_mono.pth`, `checkpoints/ablation_text_guidance_none.pth`, `checkpoints/ablation_neither.pth` — alongside the reused `checkpoints/sense_final_standard_split.pth` (the `funie_gan`/full-model arm from Phase 7).
- Deliverable: `results/ablation_master_summary.md`.
- Verification: Every ablation arm (all five newly trained checkpoints, plus the reused full-model checkpoint) has full metrics recorded; every hypothesis (H2, H3, H4) has an explicit, evidence-backed verdict; the `ablation_neither.pth` arm is listed with its Claim 1 attribution role (no hypothesis verdict).

**P10.5 — Ablation study notebook**
- Steps: Build and execute `notebooks/04_ablation_study.ipynb` (full contents in the Notebooks section below), consuming `results/experiment_3_enhancement/comparison_table.csv`, `results/experiment_4_stereo/comparison_table.csv` + `per_class_gap_analysis.md`, `results/experiment_5_text_guidance/comparison_table.csv`, `results/experiment_6_backbone_control/comparison_table.csv`, and `results/ablation_master_summary.md` from P10.1–P10.4 and P10.6 — enhancement/stereo/text-guidance ablation comparisons, backbone-matched control, full-model comparison, H2/H3/H4 verdict visualization.
- Deliverable: Executed `notebooks/04_ablation_study.ipynb`.
- Verification: Notebook runs end-to-end from already-generated P10.1–P10.4 and P10.6 outputs; no ablation training/evaluation logic duplicated inside the notebook.

**P10.6 — Backbone-matched control (supports Claim 1)**
- Steps: Train `configs/sense_neither.yaml` on the standard split with the Phase 7 schedule (EfficientNet-B4, single image, no enhancement, no text guidance). Evaluate on the standard test split with the same protocol as the other arms (macro F1, accuracy, min-class accuracy). Produce a standard-split per-scenario breakdown by grouping `standard_test.csv` samples by `scenario_id` (consistent with the Phase 10 methodology note; not a LOSO evaluation).
- Deliverable: `checkpoints/ablation_neither.pth`, `results/experiment_6_backbone_control/comparison_table.csv`.
- Verification: Macro F1, accuracy, and min-class accuracy recorded. The report states explicitly that this arm isolates backbone and training-recipe effects from SENSE's three contributions (enhancement, stereo fusion, text guidance), and that the ResNet18 baseline → SENSE-neither → full SENSE progression attributes the total gain to its sources. No hypothesis verdict — this arm supports Claim 1 only.

### Phase 10 Deliverables Checklist
- [ ] Enhancement ablation trained/evaluated (2 checkpoints: `none`, `waternet`), H2 verdict recorded, explicitly labeled as standard-split per-scenario analysis (not LOSO)
- [ ] Stereo vs. mono ablation trained/evaluated with per-class breakdown, H3 verdict recorded
- [ ] Text guidance ablation trained/evaluated with widening-gap analysis, H4 verdict recorded, explicitly labeled as standard-split per-scenario analysis (not LOSO)
- [ ] Backbone-matched control trained/evaluated (`sense_neither.yaml`), Claim 1 attribution documented
- [ ] Master ablation summary produced, all five new ablation checkpoints listed
- [ ] `notebooks/04_ablation_study.ipynb` executed with saved outputs

**Phase 10 exit criterion:** H2, H3, and H4 each have an explicit, statistically-supported verdict, positive or negative, fully documented.

---

## Phase 11 — Comparative Analysis vs. DARE & VT-UHGR

### Objectives
Produce the final head-to-head comparison anchoring the thesis results chapter, and answer RQ1 explicitly.

### Depends on
Phase 8, 9, 10 complete.

### Tasks

**P11.1 — Final comparison table**
- Steps: Rows: Yang et al. baseline — ResNet18 (Phase 4; re-run through Phase 6 Trainer with macro-F1 checkpoint selection before final reporting), SENSE-neither (EfficientNet-B4, single image, no text, no enhancement; Phase 10 P10.6), DARE (literature), VT-UHGR (literature), SENSE (Phase 7/8 standard-split result, plus the LOSO mean±SD from Phase 9). Columns: overall accuracy, macro F1, min-class accuracy, enhancement used, stereo used, text guidance used, cross-scenario tested (yes/no).
- Deliverable: `results/final_comparison_table.md`.
- Verification: Every SENSE-side number traces to a specific results file from Phases 4/8/9/10 — cited by path, never hand-typed from memory.

**P11.2 — Headline figures**
- Steps: (a) bar chart: SENSE vs. DARE vs. VT-UHGR vs. baseline, overall + macro F1 + min-class accuracy; (b) per-scenario robustness chart (Phase 9) with error bars from the statistical summary; (c) ablation summary chart (Phase 10) with H2/H3/H4 verdicts annotated. The ResNet18 baseline → SENSE-neither → full SENSE progression is included in the ablation chart to attribute the total gain between backbone/recipe effects and SENSE's three contributions.
- Deliverable: `results/figures/`, 300dpi minimum.
- Verification: Consistent color scheme and legible labels across all figures.

**P11.3 — Honest limitations section**
- Steps: Document every case where SENSE did not outperform DARE/VT-UHGR, and every hypothesis that was rejected or only partially supported. Note that SENSE runs at 224×224 — below EfficientNet-B4's native 380×380 — due to the 8 GB VRAM budget (P5.8), and that this resolution constraint applies equally to the SENSE-neither backbone-matched control.
- Deliverable: `results/limitations_and_honest_findings.md`.
- Verification: Every experiment from Phases 8–10 has an entry, positive or negative — this document directly answers RQ1 with appropriate epistemic honesty.

**P11.4 — Finalize the final-evaluation notebook with the full comparison**
- Steps: Extend `notebooks/06_final_evaluation.ipynb` (created in P8.4) with the scenario-wise and comparative sections that depend on Phase 9/10/11 results — `results/final_comparison_table.md`, Phase 9's per-scenario results, and Phase 10's ablation summary — so the notebook's final saved state matches the "Key Outputs" described in the Notebooks section below.
- Deliverable: Executed `notebooks/06_final_evaluation.ipynb` (fully finalized).
- Verification: Notebook runs end-to-end; every figure/table listed for this notebook in the Notebooks section is present in its saved output.

### Phase 11 Deliverables Checklist
- [ ] Final comparison table built, fully traceable
- [ ] Headline figures generated at print resolution
- [ ] Honest limitations section written, RQ1 explicitly answered
- [ ] `notebooks/06_final_evaluation.ipynb` finalized with full comparison sections

**Phase 11 exit criterion:** All results needed for the thesis results chapter exist as traceable files; RQ1–RQ4 each have an explicit, evidence-based answer.

---

## Phase 12 — Publication Artifacts & Thesis Writeup Support

### Objectives
Convert every raw results artifact into report-ready, publication-quality material, and formally inventory every figure/table/notebook the project produces.

### Depends on
Phase 11 complete.

### Tasks

**P12.1 — Publication artifact inventory (new, research-grade addition)**
- Objective: A single checklist confirming every figure and table a thesis committee or journal reviewer would expect actually exists, in the right place, at publication quality.
- Steps: Confirm and, if missing, generate each of the following, all saved under `results/figures/` (figures, 300dpi) or `results/tables/` (tables, both `.csv` and `.md`):

  **Figures**
  - Dataset class distribution (from Phase 2 EDA)
  - Dataset scenario distribution (from Phase 2 EDA)
  - Raw vs. enhancement preprocessing comparison (from Phase 3 QA)
  - Training loss curve (per major run: baseline, SENSE-full, each ablation)
  - Validation loss curve (same runs)
  - Validation accuracy curve (same runs)
  - Validation macro-F1 curve (same runs)
  - Learning-rate comparison (from Phase 7's sanity sweep, if performed)
  - Hyperparameter comparison plot (from Phase 7.3)
  - Confusion matrix (standard split, Phase 8)
  - Per-class performance bar chart (Phase 8)
  - Scenario-wise performance bar chart (Phase 9)
  - LOSO comparison chart with error bars (Phase 9)
  - Ablation study comparison chart (Phase 10)

  **Tables**
  - Dataset statistics table (Phase 2 EDA)
  - Hyperparameter summary table (Phase 7.3)
  - Baseline performance table (Phase 4)
  - Ablation results table (Phase 10)
  - Robustness/LOSO results table with mean±SD and CI (Phase 9)
  - Final evaluation summary table (Phase 11)

- Deliverable: `results/publication_artifact_checklist.md` mapping every item above to its exact file path and the phase/task that produced it.
- Verification: Zero unchecked items on the checklist; every path in the checklist resolves to an existing, non-empty file.

**P12.2 — Export all tables in report-ready format**
- Steps: Convert every `results/*.csv`/`.md` table into clean Markdown tables matching the formatting conventions already used in the thesis report sections built earlier (background study, novelty statement, etc.).
- Deliverable: `results/thesis_ready_tables.md`.
- Verification: Visual/formatting consistency with prior report sections.

**P12.3 — Draft the Results & Discussion section skeleton**
- Steps: Structure the skeleton around RQ1–RQ4 and the three publishable claims, with placeholders pointing to exact figures/tables from P12.1.
- Deliverable: `results/results_section_draft.md`.
- Verification: Every figure/table from the P12.1 inventory is referenced at least once; no orphaned artifacts.

**P12.4 — Archive final configs**
- Steps: Copy exact YAML configs used for every headline number into `results/archived_configs/` (immutable snapshot, separate from the evolving `configs/` working directory).
- Deliverable: `results/archived_configs/`.
- Verification: Every number in `final_comparison_table.md` traces to an archived config + checkpoint pair.

### Phase 12 Deliverables Checklist
- [ ] Publication artifact inventory complete, zero missing items
- [ ] Thesis-ready tables exported
- [ ] Results section skeleton drafted, fully cross-referenced
- [ ] Final configs archived and traceable

**Phase 12 exit criterion:** Every figure/table a thesis committee or journal reviewer would expect exists, is inventoried, and is traceable to its exact source phase/task/config/checkpoint.

---

## Phase 13 — Reproducibility & Handover

### Objectives
Ensure third-party reproducibility and proper archival, closing the loop the Evaluation Strategy and Experiment Tracking sections were built to support.

### Depends on
Phase 12 complete.

### Tasks

**P13.1 — Final README pass**
- Steps: Full project summary; reproduction instructions for every named config; evaluation reproduction instructions; `results/` directory map; explicit statement of the train/val/test protocol and why it exists (leakage prevention, tied back to the Evaluation Strategy section).
- Deliverable: Finalized `README.md`.
- Verification: A person unfamiliar with the project can reproduce Experiment 1's headline number from README instructions alone.

**P13.2 — Freeze dependency versions**
- Steps: `pip freeze > requirements-lock.txt` from the final working environment.
- Deliverable: `requirements-lock.txt`.
- Verification: A clean venv installing from this file reproduces exact package versions used for final results.

**P13.3 — Archive checkpoints externally**
- Steps: Export final checkpoints — standard-split (`sense_final_standard_split.pth`), 8 LOSO folds (`scenario_holdout_{1..8}_best.pth`), and all 5 ablation variants (`ablation_enhancement_none.pth`, `ablation_enhancement_waternet.pth`, `ablation_stereo_mono.pth`, `ablation_text_guidance_none.pth`, `ablation_neither.pth`) — to external storage with a manifest.
- Deliverable: External archive + `checkpoints/ARCHIVE_MANIFEST.md`.
- Verification: Manifest cross-references every checkpoint referenced in `final_comparison_table.md`.

**P13.4 — Reproducibility checklist (new, research-grade addition)**
- Objective: A single, explicit checklist a reviewer or examiner can use to confirm the project meets baseline reproducibility standards for a thesis/journal submission.
- Steps: Compile `results/REPRODUCIBILITY_CHECKLIST.md` confirming:
  - [ ] Random seeds fixed and documented for every reported run
  - [ ] Exact dependency versions locked (`requirements-lock.txt`)
  - [ ] Every reported number traces to an archived config + checkpoint + experiment-tracking directory
  - [ ] Train/val/test split protocol documented and leakage-checked (Phase 2)
  - [ ] Data source is the official, citable CADDY release (Phase 1)
  - [ ] Pretrained enhancement/text-encoder weights' sources documented (Phase 3, 5)
  - [ ] Statistical treatment (mean±SD, CI, paired tests) applied wherever multiple runs are aggregated (Phase 9, 10)
  - [ ] Hardware/environment specification recorded (GPU model, CUDA version, PyTorch version)
- Deliverable: `results/REPRODUCIBILITY_CHECKLIST.md`.
- Verification: Every box is checked with a linked file/path as evidence, not asserted from memory.

**P13.5 — Final repository tag**
- Steps: `git tag -a v1.0-thesis-results -m "Final results for thesis submission"`.
- Deliverable: Git tag.
- Verification: `git show v1.0-thesis-results` shows the correct final commit.

### Phase 13 Deliverables Checklist
- [ ] README finalized and reproducibility-verified
- [ ] Dependencies locked
- [ ] Checkpoints archived externally with manifest
- [ ] Reproducibility checklist completed with evidence links
- [ ] Final git tag created

**Phase 13 exit criterion:** Project is fully reproducible from a clean clone plus the external checkpoint archive, tagged as a stable final state, and passes every item on the reproducibility checklist with linked evidence.

---

## Notebooks (exploration, visualization, and analysis only — never a substitute for `scripts/`/`src/` implementation)

All notebooks live in `notebooks/`. They are strictly for exploration, visualization, debugging, and generating publication figures from results already produced by the script-based pipeline. No notebook may contain training or evaluation logic that isn't a thin call into `src/`/`scripts/` — this prevents implementation drift between "what the notebook does" and "what the pipeline actually does." `notebooks/README.md` (created in P0.8) is a brief pointer to this table; this table itself remains the authoritative list of notebooks, owners, and outputs. Each notebook file is created by the specific task named in "Produced in" below — none are created in Phase 0.

| Notebook | Produced in | Purpose | Key Outputs |
|---|---|---|---|
| `00_data_exploration.ipynb` | Phase 2 (P2.0) | Dataset statistics, class/scenario distribution, sample and stereo-pair visualization, corrupted/missing sample inspection, resolution statistics | Dataset figures/tables for `eda_findings.md` |
| `01_preprocessing.ipynb` | Phase 2/3 (P2.7, P3.6) | Raw images, augmentation pipeline visualization, enhancement (FUnIE-GAN/WaterNet) before/after comparisons, image histograms | Before/after preprocessing figures |
| `02_dataset_split_validation.ipynb` | Phase 2 (P2.8) | Split visualization, class balance per split, scenario distribution per LOSO fold, overlap/leakage verification, LOSO visualization | `split_validation_report.md` figures |
| `03_baseline_experiment_analysis.ipynb` | Phase 4 (P4.4) | Baseline training/validation loss & accuracy curves, checkpoint analysis, baseline performance summary | Baseline learning curves + summary table |
| `04_ablation_study.ipynb` | Phase 10 (P10.5) | Enhancement/stereo/text-guidance ablation comparisons, backbone-matched control (P10.6), full-model comparison, H2/H3/H4 verdict visualization | Ablation figures + comparison tables |
| `05_hyperparameter_analysis.ipynb` | Phase 7 (P7.5) | Learning-rate/batch-size/loss-weight comparison, early-stopping behavior analysis, best-configuration summary | Hyperparameter comparison plots + final config table |
| `06_final_evaluation.ipynb` | Phase 8 (P8.4), 11 (P11.4) | Confusion matrix, classification report, precision/recall/macro-F1, per-class and scenario-wise performance | Final evaluation figures/tables for thesis and paper |
| `07_robustness_analysis.ipynb` | Phase 9 (P9.5) | Scenario-wise and fold-wise performance, mean±SD/CI visualization, robustness discussion, failure-case analysis | LOSO robustness figures + statistical summary tables |

---

## Publication Artifact Index

This mirrors P12.1 in compact reference form — see Phase 12 for the full generation checklist.

**Figures:** dataset distribution · preprocessing comparison · training/validation loss curves · validation accuracy curve · validation macro-F1 curve · learning-rate comparison · hyperparameter comparison · confusion matrix · per-class performance · scenario-wise performance · LOSO comparison · ablation study comparison.

**Tables:** dataset statistics · hyperparameter summary · baseline performance · ablation results · robustness/LOSO results (mean±SD, CI) · final evaluation summary.

All saved under `results/figures/` and `results/tables/`, inventoried in `results/publication_artifact_checklist.md`.

---

## Appendix A — Config Flag Reference

| Flag | Location | Values | Effect |
|---|---|---|---|
| `data.enhancement_method` | dataset config | `none` \| `funie_gan` \| `waternet` | Which cached image variant `CaddyStereoDataset` loads |
| `model.stereo.enabled` | model config | `true` \| `false` | Whether `BiChannelEncoder` uses both branches (stereo) or left-only (mono) |
| `model.text_guidance.enabled` | model config | `true` \| `false` | Whether `SENSE` uses cross-modal fusion + cosine head, or fallback linear head |
| `model.text_guidance.encoder_type` | model config | `word2vec` \| `sentence_bert` \| `bert` | Which text-embedding backbone `GestureTextEncoder` loads; project default (`sentence_bert`) is set in `configs/sense_full.yaml`, never hardcoded |
| `model.text_guidance.encoder_name` | model config | string (e.g. `all-MiniLM-L6-v2`) | Specific pretrained model name/path passed to the selected `encoder_type` |
| `model.shared_weights` | model config | `true` \| `false` | Whether the two `BiChannelEncoder` branches share weights |
| `training.loss.alpha` / `.beta` | training config | float | CE vs. contrastive loss weighting |
| `training.loss.class_weighted` | training config | `true` \| `false` | Whether CrossEntropy in `SenseLoss` uses inverse-frequency class weights from `compute_class_weights()` (P2.9); default `true` for all SENSE runs, `false` for the Phase 4 baseline to preserve literature comparability |
| `training.phases` | training config | epoch ranges | 3-phase progressive unfreezing schedule boundaries |
| `data.split.protocol` | data config | `standard` \| `loso` | Which split family (Phase 2.2 vs 2.3) a run uses |

## Appendix B — Task ID Quick Reference

Use `P{phase}.{task}` IDs to request specific work, e.g.: "Implement P2.0" → build the EDA notebook and findings doc; "Implement P6.5" → build the experiment tracker; "Implement P9.3" → run the LOSO statistical analysis.

## Appendix C — Non-Negotiable Checkpoints

1. **End of Phase 0:** CUDA + GPU verified working.
2. **End of Phase 1:** Dataset integrity report shows zero missing-file errors.
3. **End of Phase 2:** `split_validation_report.md` shows zero leakage across the standard split and all 8 LOSO folds.
4. **End of Phase 4:** Baseline reproduction accuracy is plausible (not near-random).
5. **End of Phase 5:** All four `SENSE` mode combinations pass forward+backward tests within VRAM budget.
6. **Before any Phase 8 evaluation is trusted:** confirm the checkpoint being evaluated was selected using validation-split performance only (per the Evaluation Strategy section), and that no test split — standard or LOSO — was accessed at any point before this final evaluation step.
7. **Before any headline number is reported in the thesis/paper:** confirm it exists in `results/publication_artifact_checklist.md` with a traceable path to its source config, checkpoint, and experiment-tracking directory (per Phase 12).

## Appendix D — Final Deliverables (expanded)

- Full source code (`src/`, `scripts/`, `configs/`)
- All trained models/checkpoints (standard-split, 8 LOSO folds, 5 ablation variants: `ablation_enhancement_none.pth`, `ablation_enhancement_waternet.pth`, `ablation_stereo_mono.pth`, `ablation_text_guidance_none.pth`, `ablation_neither.pth`)
- All configuration files, including the immutable `results/archived_configs/` snapshot
- Complete experiment logs (`runs/{experiment}/{timestamp}/` — config, metrics.csv, training.log, seed, runtime, GPU info, git commit, per P6.5)
- Full TensorBoard logs
- All generated figures and tables (per the Publication Artifact Index)
- All eight analysis/exploration notebooks, executed with saved outputs
- Thesis-ready artifacts: `results/thesis_ready_tables.md`, `results/results_section_draft.md`
- `results/REPRODUCIBILITY_CHECKLIST.md` with linked evidence for every item
- `requirements-lock.txt` and full environment specification
- Finalized `README.md`
