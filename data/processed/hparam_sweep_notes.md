# Hyperparameter Sweep Notes — P7.3

Generated: 2026-11-15 · Updated: 2026-11-17 (control diagnostic completed; Attempt 2 decision pending)    
Plan reference: `Final_plan.md` §P7.3  
Tie-break rule: highest `val_macro_f1`; `val_min_class_accuracy` as tie-breaker (Evaluation Strategy section of `Final_plan.md`).

---

## Run 1 — Reference

**Config:** `configs/sense_full.yaml`  
**Run folder:** `runs/sense_full/20261114T122905Z`  
**Source file:** `runs/sense_full/20261114T122905Z/metrics.csv`

### Best checkpoint

| Field | Value |
|-------|-------|
| Best epoch (by val macro F1, min-class accuracy as tie-breaker) | **49** |
| `val_macro_f1` | 0.8766090396371016 |
| `val_accuracy` | 0.8981744421906693 |
| `val_min_class_accuracy` | 0.7704918032786885 |

### Phase-mean macro F1

| Phase | Epoch range | Mean `val_macro_f1` |
|-------|-------------|---------------------|
| Phase 1 (frozen backbone) | 1 – 10 | 0.6436452980621253 |
| Phase 2 (unfreeze last 2 blocks) | 11 – 30 | 0.7778202935041366 |
| Phase 3 (full backbone) | 31 – 50 | 0.8492740469542018 |

### Epoch-10 → Epoch-11 transition (Phase 1 → Phase 2 unfreeze)

| Epoch | `val_macro_f1` |
|-------|----------------|
| 10 | 0.7101939556612953 |
| 11 | 0.7156801205981114 |
| **Change** | **+0.0054861649368161** |

No discontinuous jump at the Phase 2 unfreeze (epoch 11). The change (+0.005) is within the normal per-epoch improvement range observed across Phase 1.

### Mean epoch wall time per phase

| Phase | Epoch range | Mean epoch time (s) |
|-------|-------------|---------------------|
| Phase 1 | 1 – 10 | 150.933 |
| Phase 2 | 11 – 30 | 158.431 |
| Phase 3 | 31 – 50 | 245.423 |

---

## Variation 1 — Higher base LR

**Config:** `configs/sense_full_lr1e-3.yaml`

### What was changed

| Parameter | Old value | New value |
|-----------|-----------|-----------|
| `training.base_lr` | 1.0e-04 | 1.0e-03 |

All other hyperparameters (alpha, beta, batch_size, image_size, phase boundaries, patience,
weight_decay, scheduler_eta_min_ratio, lr_scale per phase) are inherited unchanged from
`sense_full.yaml` via `_base_: sense_full.yaml`.

### Why it was changed

Three symptoms observed in Run 1's `metrics.csv` motivate the increase:

1. **No unfreeze jump at epoch 11.** The change in `val_macro_f1` from epoch 10 to epoch 11
   was only +0.0055 — indistinguishable from a normal inter-epoch step. The head LR appears
   low enough that unfreezing brought no acceleration.
2. **Macro F1 still rising at epoch 50 (the LR floor).** Phase 3 mean `val_macro_f1` (0.8493)
   exceeds Phase 2 mean (0.7778) and the best checkpoint is epoch 49 — the model was still
   improving when the cosine schedule hit its floor, indicating the run was cut short by the
   epoch budget rather than by the learning signal.
3. **Head and backbone LRs are both 10× lower than a typical fine-tuning baseline**
   (≈ 9.0e-06 at the start of Phase 2, ≈ 3.2e-07 at the start of Phase 3).

---

### Attempt 1 — 2026-11-15 — DIVERGED

**Run folder:** `runs/sense_full_lr1e-3/20261115T142011Z` (14 epochs, no healthy best.pth)
**Status:** failed — train losses NaN from epoch 6 onward.

| Epoch | `train_loss` | `val_loss` | `val_accuracy` | `val_macro_f1` |
|-------|--------------|------------|----------------|----------------|
| 5 (last healthy) | 1.4649 | 1.2736 | 0.6422 | 0.5463 |
| 6 | **nan** | **nan** | 0.5805 | 0.4800 |
| 7 – 14 | nan | nan | **0.11075 (constant)** | **0.01173 (constant)** |

The epoch 7+ signature (constant val_accuracy 0.11075 = single-class predictions from
all-NaN logits; constant macro F1 0.01173) shows the model was fully dead after epoch 6.
Epoch 6 itself was partially broken (val losses NaN but accuracy still meaningful),
indicating corruption began mid-epoch-6.

**Diagnosis (leading hypothesis C):** fp16 forward overflow upstream in the backbone
under autocast + BatchNorm running-stat buffer corruption (buffers update during forward
even while the backbone is frozen, outside GradScaler's step-skip protection) under the
10× LR. GradScaler silently skipped the overflowing weight updates, masking the instability
until the corrupted BN buffers poisoned every subsequent forward pass.

**Investigation (training-step audit, 4 items — all closed):**
1. AMP order verified correct (`scale → backward → unscale_ → clip → step → update`);
   ordering regression test added.
2. Head/loss dtypes audited empirically on CUDA — fp16 overflow in the cosine head or
   losses **disproven** (softmax/exp compute in fp32; temperature clamp-bounded);
   fp32 hardening applied anyway by construction (`autocast(enabled=False)` regions).
3. Fail-fast divergence guard added: 20-consecutive-non-finite-step abort, non-finite
   val-loss abort before checkpoint saves, finiteness gates on best.pth AND last.pth.
4. Per-epoch mean unscaled grad norm + clipping fraction logged to training.log.

**Evidence gap:** no gradient statistics exist for Attempt 1 — item 4 was implemented
after this run. The CSV signature above is the only record of its failure dynamics.

---

### Attempt 2 — post-hardening rerun — PENDING

**Status:** pending — the `sense_neither.yaml` control diagnostic completed 2026-11-17
(see the section below): the pipeline trains fully healthily at lr 1e-4 with the guard
active, and its gradient regime is now characterized. The remaining decision is between
rerunning lr1e-3 with the guard (Attempt 2 as specified below), going directly to an
intermediate LR (3e-4), or adding warmup.

**Protocol for Attempt 2 (differs from Attempt 1):**
- Divergence guard active: aborts within ~20 consecutive non-finite steps (~1 minute)
  instead of limping for 8 dead epochs; last.pth preserved at the last healthy epoch.
- Per-epoch gradient stats in training.log provide the evidence Attempt 1 lacks:
  climbing mean norm + high clipping fraction before a NaN supports hypothesis C;
  low clipping with NaN appearing refutes it.
- fp32-hardened head and losses remove the overflow-sensitive ops from the equation.

**Outcome:** pending.
**Decision:** pending — will be recorded here after Attempt 2's metrics.csv (and, if it
diverges, its training.log gradient story) is available.

---

## Backbone-matched control diagnostic — sense_neither — 2026-11-17 — COMPLETED

Not a sweep variation. Dual purpose: (a) diagnostic that the pipeline trains healthily
at lr 1e-4 with the post-audit guard active, (b) the Phase 10 P10.6 backbone-matched
control artifact.

**Config:** `configs/sense_neither.yaml` (inherits `sense_full.yaml`; stereo off,
text guidance off, enhancement none — everything else identical to Run 1)
**Run folder:** `runs/sense_neither/20261117T133503Z` (50/50 epochs, zero errors,
zero guard firings, no early stop)

### Best checkpoint

| Field | Value |
|-------|-------|
| Best epoch (by val macro F1) | **45** |
| `val_macro_f1` | 0.8287 |
| `val_accuracy` | 0.8576 |
| `val_min_class_accuracy` | 0.6566 |
| `val_loss` | 0.5397 |

Final epoch 50 (lr floor 1e-6): macro F1 0.8108, val_acc 0.8507 — model plateaued,
oscillating ~0.81–0.83 through the LR floor.

### Control vs Run 1 (identical hyperparameters and seed; only SENSE modules differ)

| Metric (best checkpoint) | sense_neither (control) | Run 1 (full SENSE) | SENSE contribution |
|--------------------------|-------------------------|--------------------|--------------------|
| `val_macro_f1` | 0.8287 | 0.8766 | **+0.048** |
| `val_accuracy` | 0.8576 | 0.8982 | +0.041 |
| `val_min_class_accuracy` | 0.6566 | 0.7705 | **+0.114** |
| Best epoch | 45 | 49 | — |

### Gradient-regime characterization (first complete gradient history for this pipeline)

- Phase 1: mean unscaled grad norm ~13.5, clipping active in 100% of steps
  (natural scale ≈ 13× the clip ceiling of 1.0).
- Phase 2: norm declined steadily to ~8.7, clipping 98–99% — healthy settling trend.
- Phase 3: norm bumped to a stable ~11–12 at the unfreeze, clip fraction ~99%,
  no pegging through epoch 50.
- `inf` means occurred every ~2000 steps, matching GradScaler's growth_interval —
  routine scale-probe overflows, silently self-corrected.
- Occasional 1–2 NaN-gradient steps per epoch, silently skipped by the scaler;
  losses stayed finite throughout, guard never fired.

### Findings

1. **Pipeline fully healthy at lr 1e-4** — epoch 6 (Attempt 1's death epoch) passed
   clean; divergence is LR-specific, consistent with hypothesis C.
2. **The pipeline natively operates in a clipped regime** (norm ~9–13 vs clip 1.0,
   ~99% clipping even at convergence) — relevant context for interpreting Attempt 2's
   gradient story at 10× LR.
3. **Unfreeze-response asymmetry:** the control shows clear macro-F1 jumps at both
   unfreezes (+0.024 at Phase 2, +0.026 at Phase 3); Run 1's recorded Phase 2
   transition was +0.0055 (no jump). The SENSE modules appear to smooth optimization
   across phase transitions.
4. **SENSE contribution quantified:** +4.8 macro F1, +4.1 accuracy, +11.4 min-class
   accuracy under identical conditions — the P10.6 ablation evidence.

---

### How the winner is selected

Per the Evaluation Strategy in `Final_plan.md`: the configuration with the **highest
`val_macro_f1`** at its best checkpoint wins. If two configurations tie on `val_macro_f1`
(within the 0.005 tie-break window defined in `src/evaluation/metrics.py::is_better_checkpoint`),
`val_min_class_accuracy` is the tie-breaker. Selection is made from the saved `best.pth`
checkpoints only — never from the last epoch — and the winning configuration is named
explicitly in writing before any test-split evaluation is performed.

### What evidence will be generated

- A comparison plot of `val_macro_f1` vs. epoch for Run 1 and the surviving Variation 1
  attempt, saved to `results/hparam_sweep/comparison_plot.png`.
- A summary table saved to `results/hparam_sweep/summary_table.csv` with one row per run:
  config, best epoch, best `val_macro_f1`, best `val_accuracy`, best
  `val_min_class_accuracy`, phase-mean macro F1 (phases 1–3), and total wall time.
- The winning configuration stated in writing in this file under a "Decision" section,
  with justification referencing the above evidence.