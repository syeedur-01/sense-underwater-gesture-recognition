# Cross-scenario (LOSO) statistical summary

Folds reported: 8 of 8.

Per-fold macro F1 is averaged over the classes that the fold's test partition
contains (D21). The seventeen-class value is reported beside it, together with
class coverage, because five of the eight held-out scenarios do not record every
gesture and a seventeen-class macro F1 is therefore capped by coverage rather than
by model quality.

Each fold's held-out scenario was read once, after that fold's checkpoint was frozen by the training loop. The same evaluation entry point was used for every fold, and for the standard test partition.

## Aggregate across folds

| Metric | Folds | Mean | SD | 95% CI for the mean |
|---|---:|---:|---:|---|
| Accuracy | 8 | 0.6542 | 0.1546 | [0.5250, 0.7834] |
| Macro F1 (present classes) | 8 | 0.5060 | 0.2130 | [0.3280, 0.6841] |
| Macro F1 (all 17 classes) | 8 | 0.3898 | 0.1765 | [0.2423, 0.5374] |
| Minimum class accuracy | 8 | 0.1380 | 0.2819 | [-0.0977, 0.3737] |

The confidence interval is computed from the t distribution with n - 1 degrees of
freedom. With eight folds it is wide; it is reported as a description of spread
across scenarios, not as an estimate of run-to-run variability, which is the
single-seed limitation recorded separately.

## Per-fold results

| Fold | Held-out scenario | Test pairs | Classes present | Accuracy | Macro F1 (present) | Macro F1 (17) | Min class accuracy |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | biograd-A | 3,041 | 17/17 | 0.7701 | 0.4952 | 0.4952 | 0.0000 |
| 2 | biograd-B | 1,772 | 17/17 | 0.6828 | 0.4969 | 0.4969 | 0.0000 |
| 3 | biograd-C | 3,292 | 15/17 | 0.7682 | 0.6822 | 0.6019 | 0.2606 |
| 4 | brodarski-A | 447 | 8/17 | 0.4452 | 0.3010 | 0.1416 | 0.0000 |
| 5 | brodarski-B | 157 | 10/17 | 0.5350 | 0.2189 | 0.1288 | 0.0000 |
| 6 | brodarski-C | 985 | 10/17 | 0.4893 | 0.5686 | 0.3345 | 0.0441 |
| 7 | brodarski-D | 628 | 10/17 | 0.8885 | 0.8883 | 0.5226 | 0.7993 |
| 8 | genova-A | 6,107 | 17/17 | 0.6543 | 0.3974 | 0.3974 | 0.0000 |

## Class coverage

Mean coverage across folds: 0.7647 of the seventeen
classes. Folds with incomplete coverage are scored on what they contain; the
absent classes are listed per fold in `per_scenario_results.csv`.

The smallest fold is `brodarski-B` with 157 test pairs, and the largest is `genova-A` with 6,107. The smallest fold
contributes most of the variance across folds and is reported with its sample
size attached.
