# Data Split Validation & Anti-Leakage Audit Report

**Phase 2 Exit Deliverable | SENSE Project**

---

## 1. Executive Summary

- **Total Audited Stereo Pairs:** **16,429** (`synthetic == 0` real images only)
- **Standard 70/15/15 Split Status:** **PASS** (Zero sample_id overlap)
- **LOSO 8-Fold Cross-Scenario Status:** **PASS** (All 8 folds zero leakage)
- **Overall Anti-Leakage Audit:** **✓ ALL CHECKS PASSED PERFECTLY**

---

## 2. Standard 70/15/15 Stratified Split Audit

| Split | Sample Count | Proportion | Pairwise Overlap Check | Status |
|---|---|---|---|---|
| `standard_train.csv` | **11,499** | 70.0% | train ∩ val = **0** | ✓ PASS |
| `standard_val.csv` | **2,465** | 15.0% | train ∩ test = **0** | ✓ PASS |
| `standard_test.csv` | **2,465** | 15.0% | val ∩ test = **0** | ✓ PASS |

**Per-Class Proportion Preservation:** Maximum per-class percentage difference across standard train/val/test splits is **0.0244%**, far below the 2.0% requirement threshold.

---

## 3. Leave-One-Scenario-Out (LOSO) 8-Fold Audit

| Fold | Held-Out Scenario | Train Count | Val Count | Test Count | Audit Status |
|---|---|---|---|---|---|
| Fold 1 | `biograd-A` | 11,379 | 2,009 | 3,041 | ✓ PASS |
| Fold 2 | `biograd-B` | 12,458 | 2,199 | 1,772 | ✓ PASS |
| Fold 3 | `biograd-C` | 11,166 | 1,971 | 3,292 | ✓ PASS |
| Fold 4 | `brodarski-A` | 13,584 | 2,398 | 447 | ✓ PASS |
| Fold 5 | `brodarski-B` | 13,831 | 2,441 | 157 | ✓ PASS |
| Fold 6 | `brodarski-C` | 13,127 | 2,317 | 985 | ✓ PASS |
| Fold 7 | `brodarski-D` | 13,430 | 2,371 | 628 | ✓ PASS |
| Fold 8 | `genova-A` | 8,773 | 1,549 | 6,107 | ✓ PASS |

### Programmatic Invariants Verified Per Fold:
1. `test` set contains **exclusively** the designated held-out scenario (zero scenario contamination).
2. `train` ∪ `val` sets contain **zero** samples from the held-out scenario.
3. `train` ∩ `val` is **empty** (zero sample_id overlap).

---

## 4. Phase 2 Exit Criteria Verification

| Criterion | Requirement | Result |
|---|---|---|
| Dataset Verification | Zero missing files, 8 scenarios, 16 gesture classes | ✓ PASSED |
| EDA Findings | Document class & scenario imbalance before split | ✓ PASSED ([`eda_findings.md`](file:///c:/Users/Tamim%20Syeedur%20Rahman/All%20Codes/sense-project/data/processed/eda_findings.md)) |
| Data Manifest | `manifest.csv` built with 16,429 rows | ✓ PASSED |
| Anti-Leakage Audit | Zero sample_id overlap across all splits | ✓ PASSED |
| PyTorch Dataset & DataLoader | `CaddyStereoDataset`, stereo transforms, `get_dataloader()` | ✓ PASSED |
| Unit Tests | `pytest tests/test_data.py` passing | ✓ PASSED (4/4 tests pass) |

**Conclusion:** Progression to **Phase 3 (Enhancement Module Integration)** is fully authorized.
