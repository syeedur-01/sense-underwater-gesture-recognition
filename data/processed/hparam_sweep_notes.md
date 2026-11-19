# Hyperparameter Sweep Notes — P7.3

Generated: 2026-11-15 · Updated: 2026-11-19 (Variations 1–3 recorded; winner declared)    
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

### Attempt 2 — post-hardening rerun — DIVERGED (guard-confirmed)

**Run folder:** `runs/sense_full_lr1e-3/20261117T162825Z`
**Status:** aborted by the divergence guard at epoch 5, batch 243 — 20 consecutive
non-finite training-loss steps (tolerance 20). Total wall time 645.72 s; the abort fired
46 s into epoch 5.

**Protocol applied:** divergence guard active, fp32-hardened head and losses, per-epoch
gradient stats logged to `training.log`.

| Epoch | `train_loss` | `val_loss` | `val_accuracy` | `val_macro_f1` |
|-------|--------------|------------|----------------|----------------|
| 1 | 2.3981 | 2.0219 | 0.3205 | 0.2272 |
| 2 | 1.9947 | 1.7788 | 0.4000 | 0.3097 |
| 3 | 1.8800 | 1.8798 | 0.3870 | 0.2730 |
| 4 | **nan** (32 non-finite batches skipped) | 1.7181 | 0.4215 | 0.3029 |
| 5 | — aborted at batch 243 | — | — | — |

First non-finite training loss: epoch 4, batch 435.

**Checkpoints preserved:**

| Checkpoint | Epoch | `val_macro_f1` | `val_accuracy` | `val_min_class_accuracy` |
|-----------|-------|----------------|----------------|--------------------------|
| `best.pth` | 2 | 0.3097 | 0.4000 | 0.0000 |
| `last.pth` | 4 | 0.3029 | 0.4215 | 0.0000 |

Both verified loadable; no checkpoint holds a diverged state.

**Gradient evidence (Attempt 1 had none):**

| Epoch | Mean unscaled grad norm | Clipping active |
|-------|------------------------|-----------------|
| 3 | 0.6923 | 13.5% (97/719) |
| 4 | 0.6640 | 11.4% (78/687) |

For reference, the healthy 3e-4 run operates at mean norm 2.5–4.6 with clipping active in
90–100% of steps. At 1e-3 the gradients were **smaller**, not larger, and clipping rarely
engaged — yet the loss still went non-finite. That refutes the gradient-explosion reading
and supports hypothesis C: corruption originates in the forward pass (fp16 overflow), not
the backward pass. `val_loss` remained finite through epoch 4, confining the failure to
training-side computation.

**Conclusion:** 1e-3 is outside this pipeline's stability region, confirmed independently
twice (Attempt 1 at epoch 6, Attempt 2 at epochs 4–5). No further 1e-3 attempts.

---

### Attempt 3 — intermediate base LR (3e-4) — COMPLETED

**Config:** `configs/sense_full_lr3e-4.yaml`
**Run folder:** `runs/sense_full_lr3e-4/20261117T172308Z` (50/50 epochs, 10,284.67 s ≈ 2 h 51 m,
zero non-finite batches skipped, zero guard firings, no early stop)

#### What was changed

| Parameter | Old value | New value |
|-----------|-----------|-----------|
| `training.base_lr` | 1.0e-04 | 3.0e-04 |

Half-log step between Run 1 (1e-4) and the twice-fatal 1e-3. All other values inherited
unchanged from `sense_full.yaml`.

#### Why it was changed

1e-3 failed twice, but Run 1's metrics showed the model was still improving at the epoch-50
LR floor, so 1e-4 was likely too low. 3e-4 is the geometric midpoint of the two.

#### Best checkpoint

| Field | Value |
|-------|-------|
| Best epoch | **41** |
| `val_macro_f1` | 0.894619 |
| `val_accuracy` | 0.916836 |
| `val_min_class_accuracy` | 0.827586 |
| `val_loss` | 0.625926 |

Final epoch 50: macro F1 0.8830 at the LR floor (3e-6).

#### Phase-mean macro F1

| Phase | Epoch range | Mean `val_macro_f1` | Mean epoch time (s) |
|-------|-------------|---------------------|---------------------|
| Phase 1 (frozen backbone) | 1 – 10 | 0.587017 | 156.5 |
| Phase 2 (unfreeze last 2 blocks) | 11 – 30 | 0.799992 | 176.3 |
| Phase 3 (full backbone) | 31 – 50 | 0.876272 | 256.5 |

#### Unfreeze transitions

| Transition | Δ `val_macro_f1` |
|------------|------------------|
| Epoch 10 → 11 (Phase 2 unfreeze) | **+0.084988** |
| Epoch 30 → 31 (Phase 3 unfreeze) | **+0.026001** |

Both unfreezes now produce clear jumps — the symptom that motivated Variation 1 is resolved
at 3e-4 (Run 1's Phase 2 transition was only +0.0055).

#### vs Run 1 (1e-4)

| Metric | Run 1 | Attempt 3 (3e-4) | Δ |
|--------|-------|------------------|---|
| `val_macro_f1` | 0.876609 | **0.894619** | **+0.018010** |
| `val_accuracy` | 0.898174 | 0.916836 | +0.018662 |
| `val_min_class_accuracy` | 0.770492 | 0.827586 | +0.057094 |
| Best epoch | 49 | 41 | — |

**Outcome:** completed with no instability. `val_macro_f1` +0.0180 over Run 1 — outside the
0.005 tie window. Selected as the LR baseline for subsequent variations.

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

## Variation 2 — Phase-3 backbone LR scale

**Config:** `configs/sense_full_lr3e-4_p3lr0.1.yaml`
**Run folder:** `runs/sense_full_lr3e-4_p3lr0.1/20261119T063459Z` (50/50 epochs,
11,411.95 s ≈ 3 h 10 m, zero non-finite batches skipped, zero guard firings, no early stop)

### What was changed

| Parameter | Old value | New value |
|-----------|-----------|-----------|
| `training.phases.phase3.lr_scale` | 0.01 | 0.1 |

Single axis. Inherits `sense_full_lr3e-4.yaml` via `_base_`; base_lr, phase boundaries,
unfreeze schedule, alpha/beta, class_weighted, weight_decay and scheduler are all unchanged.

`lr_scale` multiplies `base_lr` for the **backbone parameter group only** — head, fusion,
attention and text-encoder parameters always train at `base_lr`
(`src/training/trainer.py::_build_optimizer_for_phase`). Effect in phase 3:

| Group | Before | After |
|-------|--------|-------|
| Head / fusion / attention | 3.0e-04 | 3.0e-04 (unchanged) |
| Backbone (all blocks) | 3.0e-06 | 3.0e-05 |

### Why it was changed

Phase 3 is the only phase in which all backbone blocks are unfrozen, and at 0.01 the
backbone learned at 3e-6 — 33× below the 1e-4 at which the ResNet18 baseline fine-tunes its
entire network, and 100× below the head's own LR. Under that ratio the pretrained
EfficientNet features can barely adapt while the cosine head carries the entire fit. 0.1
gives 3e-5: 10× below the head and 3× below the baseline, so it stays well inside the
stability region that 1e-3 violated.

### Best checkpoint

| Field | Value |
|-------|-------|
| Best epoch | **45** |
| `val_macro_f1` | 0.929634 |
| `val_accuracy` | 0.950101 |
| `val_min_class_accuracy` | 0.857143 |
| `val_loss` | 0.508554 |

Final epoch 50: macro F1 0.9261 at the LR floor (3e-6).

**Note on checkpoint selection:** the highest single-epoch macro F1 in `metrics.csv` is
epoch 46 at 0.934241 (acc 0.951318, min-class 0.800000). It was not saved as `best.pth`
because +0.0046 over epoch 45's 0.929634 falls inside the 0.005 tie window, and the
tie-breaker (min-class accuracy) favoured epoch 45 (0.857143 vs 0.800000). This is the
documented selection rule (`is_better_checkpoint`) operating as designed.

### Phase-mean macro F1

| Phase | Epoch range | Mean `val_macro_f1` | Mean epoch time (s) |
|-------|-------------|---------------------|---------------------|
| Phase 1 (frozen backbone) | 1 – 10 | 0.574852 | 186.1 |
| Phase 2 (unfreeze last 2 blocks) | 11 – 30 | 0.797725 | 182.4 |
| Phase 3 (full backbone) | 31 – 50 | 0.915468 | 291.6 |

### Unfreeze transitions

| Transition | Δ `val_macro_f1` |
|------------|------------------|
| Epoch 10 → 11 (Phase 2 unfreeze) | +0.035078 |
| Epoch 30 → 31 (Phase 3 unfreeze) | +0.032743 |

### vs Variation 1 Attempt 3 (the only changed axis)

| Metric | V1-A3 (`lr_scale` 0.01) | V2 (`lr_scale` 0.1) | Δ |
|--------|------------------------|---------------------|---|
| `val_macro_f1` | 0.894619 | **0.929634** | **+0.035015** |
| `val_accuracy` | 0.916836 | 0.950101 | +0.033265 |
| `val_min_class_accuracy` | 0.827586 | 0.857143 | +0.029557 |
| Phase-1 mean `val_macro_f1` | 0.587017 | 0.574852 | −0.012165 |
| Phase-2 mean `val_macro_f1` | 0.799992 | 0.797725 | −0.002267 |
| Phase-3 mean `val_macro_f1` | 0.876272 | 0.915468 | **+0.039196** |

**Attribution:** phases 1–2 have identical configs in both runs and land within run-to-run
noise of each other; the entire gain appears in phase 3, the only phase whose LR scale
changed. The phase-3 mean of this run (0.9155) exceeds the previous run's best single epoch
(0.8946). Phase-3 epoch time rose from ~256 s to ~292 s — the expected cost of the full
backbone actually receiving gradients.

---

## Variation 3 — Phase-3 backbone LR scale, second step

**Config:** `configs/sense_full_lr3e-4_p3lr0.3.yaml`
**Run folder:** `runs/sense_full_lr3e-4_p3lr0.3/20261119T120231Z` (50/50 epochs,
11,640.19 s ≈ 3 h 14 m, zero non-finite batches skipped, zero guard firings, no early stop)

### What was changed

| Parameter | Old value | New value |
|-----------|-----------|-----------|
| `training.phases.phase3.lr_scale` | 0.1 | 0.3 |

Single axis; inherits `sense_full_lr3e-4_p3lr0.1.yaml`. `lr_scale` again applies to the
backbone parameter group only — head, fusion, attention and text encoder stay at `base_lr`:

| Group | V2 | V3 |
|-------|----|----|
| Head / fusion / attention | 3.0e-04 | 3.0e-04 (unchanged) |
| Backbone, nominal (`base_lr × lr_scale`) | 3.0e-05 | 9.0e-05 |
| Backbone, actual at epoch 31 (cosine at ≈0.32 of peak) | ≈9.7e-06 | ≈2.9e-05 |

### Why it was changed

V2's response to the first step (0.01 → 0.1) was large (+0.035015) and confined to phase 3,
so the axis was clearly live. This step tests whether that response is still rising. 0.3 puts
the backbone at a nominal 9e-5 — just under the 1e-4 at which the ResNet18 baseline fine-tunes
its entire network — while remaining 3.3× below the head LR and ≈11× below the 1e-3 that
diverged twice.

### Best checkpoint

| Field | Value |
|-------|-------|
| Best epoch | **45** |
| `val_macro_f1` | 0.956383 |
| `val_accuracy` | 0.965923 |
| `val_min_class_accuracy` | 0.914286 |
| `val_loss` | 0.467379 |

Final epoch 50: macro F1 0.9449 at the LR floor (3e-6). Unlike V2, the `metrics.csv` peak and
the saved `best.pth` coincide (epoch 45), so no tie-break was involved.

### Phase-mean macro F1

| Phase | Epoch range | Mean `val_macro_f1` | Mean epoch time (s) |
|-------|-------------|---------------------|---------------------|
| Phase 1 (frozen backbone) | 1 – 10 | 0.576761 | 166.4 |
| Phase 2 (unfreeze last 2 blocks) | 11 – 30 | 0.782731 | 179.0 |
| Phase 3 (full backbone) | 31 – 50 | 0.934816 | 316.1 |

### Unfreeze transitions

| Transition | Δ `val_macro_f1` |
|------------|------------------|
| Epoch 10 → 11 (Phase 2 unfreeze) | +0.037452 |
| Epoch 30 → 31 (Phase 3 unfreeze) | **+0.065212** |

### vs Variation 2 (the only changed axis)

| Metric | V2 (`lr_scale` 0.1) | V3 (`lr_scale` 0.3) | Δ |
|--------|---------------------|---------------------|---|
| `val_macro_f1` | 0.929634 | **0.956383** | **+0.026749** |
| `val_accuracy` | 0.950101 | 0.965923 | +0.015822 |
| `val_min_class_accuracy` | 0.857143 | 0.914286 | +0.057143 |
| Phase-1 mean `val_macro_f1` | 0.574852 | 0.576761 | +0.001909 |
| Phase-2 mean `val_macro_f1` | 0.797725 | 0.782731 | −0.014994 |
| Phase-3 mean `val_macro_f1` | 0.915468 | 0.934816 | **+0.019348** |

### Attribution and response trend

Phases 1–2 again have identical configs and differ only within run-to-run noise; the gain is
confined to phase 3. The within-run phase-3 gain (best − epoch 30) rises monotonically along
the axis:

| Run | `lr_scale` | Phase-3 gain (best − ep 30) | Phase-3 mean |
|-----|-----------|-----------------------------|--------------|
| V1-A3 | 0.01 | +0.0502 | 0.876272 |
| V2 | 0.1 | +0.0818 | 0.915468 |
| V3 | 0.3 | **+0.1353** | **0.934816** |

Phase-3 *mean* increments are shrinking (+0.039196 → +0.019348) — diminishing returns.
Phase-3 epoch time rose again (292 s → 316 s), consistent with the backbone training harder.

### Noise consideration

Single-seed run-to-run variance is measurable directly from the three runs that share an
identical phase-1/2 recipe: their phase-2 means over epochs 11–20 are 0.7658 / 0.7613 / 0.7465,
a spread of ≈0.019 — larger than the 0.005 tie window. The decision rule for this variation was
therefore fixed before the result was seen: a delta beyond ±0.020 counts as real, inside it as
a tie. V3's +0.026749 exceeds that band.

---

## Decision — 2026-11-19

**Winning configuration:** `configs/sense_full_lr3e-4_p3lr0.3.yaml`
**Winning run:** `runs/sense_full_lr3e-4_p3lr0.3/20261119T120231Z`
**Best checkpoint:** `checkpoints/best.pth`, epoch 45

| Rank | Configuration | Best epoch | `val_macro_f1` | `val_accuracy` | `val_min_class_accuracy` |
|------|---------------|-----------|----------------|----------------|--------------------------|
| 1 | `sense_full_lr3e-4_p3lr0.3` (V3) | 45 | **0.956383** | 0.965923 | 0.914286 |
| 2 | `sense_full_lr3e-4_p3lr0.1` (V2) | 45 | 0.929634 | 0.950101 | 0.857143 |
| 3 | `sense_full_lr3e-4` (V1-A3) | 41 | 0.894619 | 0.916836 | 0.827586 |
| 4 | `sense_full` (Run 1) | 49 | 0.876609 | 0.898174 | 0.770492 |

Margin over the runner-up: **+0.026749** — above both the 0.005 tie window and the ≈0.019
single-seed run-to-run noise band, so no tie-breaker is required. Cumulative improvement over
Run 1: **+0.079774** macro F1.

The sweep is closed at three variations (P7.3 cap: 3–4). No further single-axis runs were
scheduled: phase-3 mean increments are diminishing (+0.039 → +0.019) and the remaining headroom
against the baseline is within noise.

Selected from saved `best.pth` checkpoints only; the test split was not used at any point in
the sweep. The winner is named here in writing before any test-split evaluation, per the
Evaluation Strategy in `Final_plan.md`.

---

### How the winner is selected

Per the Evaluation Strategy in `Final_plan.md`: the configuration with the **highest
`val_macro_f1`** at its best checkpoint wins. If two configurations tie on `val_macro_f1`
(within the 0.005 tie-break window defined in `src/evaluation/metrics.py::is_better_checkpoint`),
`val_min_class_accuracy` is the tie-breaker. Selection is made from the saved `best.pth`
checkpoints only — never from the last epoch — and the winning configuration is named
explicitly in writing before any test-split evaluation is performed.

### What evidence will be generated

- A comparison plot of `val_macro_f1` vs. epoch across every tried run — Run 1, Variation 1
  Attempts 1–3 (the two `lr 1e-3` divergences are drawn dashed), Variation 2 and
  Variation 3 — saved to `results/hparam_sweep/comparison_plot.png`. A second panel zooms on
  epochs 25–50 for the completed runs. *(Generated 2026-11-19 via
  `scripts/plot_hparam_sweep.py`.)*
- A summary table saved to `results/hparam_sweep/summary_table.csv` with one row per run:
  config, best epoch, best `val_macro_f1`, best `val_accuracy`, best
  `val_min_class_accuracy`, phase-mean macro F1 (phases 1–3), and total wall time; plus
  `run_dir`, `status` (completed / diverged) and the best-checkpoint `val_loss` for
  traceability. Completed runs are ranked; diverged runs are listed unranked. *(Generated
  2026-11-19 via `scripts/plot_hparam_sweep.py`.)*
- The winning configuration stated in writing in this file under a "Decision" section,
  with justification referencing the above evidence. *(Done — see "Decision — 2026-11-19".)*