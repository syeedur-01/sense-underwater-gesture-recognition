# CADDY Dataset Schema

> Generated during Phase 1 (P1.4) of the SENSE project.  
> Last updated: 2026-07-30  
> Source: `CADDY_gestures_complete_v2_release.zip` (official CADDIAN release v2)

---

## Overview

The CADDY (Cognitive Autonomous Diving Buddy) dataset is an underwater
diver-gesture recognition dataset captured using a stereo-camera rig across
8 distinct diving scenarios. All stereo pairs are labelled with one of
16 positive gesture classes **or** a no-gesture (negative) class.

| Property | Value |
|---|---|
| Total real stereo pairs | 16,429 |
| Real positive stereo pairs | 9,239 |
| Real negative stereo pairs | 7,190 |
| Positive gesture classes | 16 (label_id 0–15) |
| No-gesture class | 1 (label_id −1, negatives CSV) |
| Total model output classes | 17 |
| Scenario folders | 8 |
| Image format | JPEG (.jpg) |
| Stereo pair naming convention | `{scenario}_{nnnnn}_left.jpg` / `{scenario}_{nnnnn}_right.jpg` |

> **Synthetic variants**: Each real stereo pair has 9 synthetic distorted
> copies (blur × 3 dirs, noise × 3 dirs, colour × 3 dirs), giving 92,390
> rows in the positives CSV and 71,900 rows in the negatives CSV total.
> The SENSE pipeline uses **real images only** (`synthetic == 0`).

---

## Extracted Folder Layout

```
data/raw/caddy/
├── CADDY_gestures_complete_v2_release/          ← main archive extracted here
│   ├── CADDY_gestures_all_true_positives_release_v2.csv   ← CANONICAL CSV
│   ├── CADDY_gestures_all_true_negatives_release_v2.csv   ← CANONICAL CSV
│   ├── biograd-A/
│   │   ├── true_positives/
│   │   │   └── raw/              ← real stereo images
│   │   │       ├── biograd-A_00000_left.jpg
│   │   │       ├── biograd-A_00000_right.jpg
│   │   │       └── ...
│   │   └── true_negatives/
│   │       └── raw/
│   │           ├── biograd-A_00000_left.jpg
│   │           └── ...
│   ├── biograd-B/    (same structure)
│   ├── biograd-C/
│   ├── brodarski-A/
│   ├── brodarski-B/
│   ├── brodarski-C/
│   ├── brodarski-D/
│   └── genova-A/
├── hands_rois/                                  ← supplementary, extracted separately
│   └── ...
├── CADDY_gestures_complete_v2_release.zip       ← original archive (kept for reference)
└── hands-ROIs.zip                               ← supplementary archive (kept for reference)
```

**Important**: The CSV paths reference images relative to `CADDY_ROOT` using
Unix-style paths starting with a leading `/`. For example:
```
/biograd-A/true_positives/raw/biograd-A_00000_left.jpg
```
To resolve to an absolute disk path: `CADDY_ROOT + csv_path_field`.

---

## CSV Column Reference

### Positives CSV — `CADDY_gestures_all_true_positives_release_v2.csv`

| Column | Type | Description |
|---|---|---|
| `index` | int | Row index |
| `scenario` | str | Scenario folder name (e.g., `biograd-A`) |
| `stereo left` | str | Relative path to left image (relative to `CADDY_ROOT`) |
| `stereo right` | str | Relative path to right image |
| `label name` | str | Gesture class name (see gesture table below) |
| `label id` | int | Gesture class ID (0–15) |
| `roi left` | str | Bounding box `[x,y,w,h]` in left image |
| `roi right` | str | Bounding box `[x,y,w,h]` in right image |
| `synthetic` | int | 0 = real image; 1 = synthetically distorted copy |
| `iqa_mdm_entropy` | float | Image quality metric (real images only) |
| `iqa_mdm_d` | float | Image quality metric (real images only) |
| `iqa_mdm_dcomp` | float | Image quality metric (real images only) |
| `distortion` | str | Distortion type for synthetic images (`blur`, `channel noise`, etc.) |
| `param 1` | int/float | Distortion parameter 1 (synthetic only) |
| `param 2` | int/float | Distortion parameter 2 (synthetic only) |

### Negatives CSV — `CADDY_gestures_all_true_negatives_release_v2.csv`

| Column | Type | Description |
|---|---|---|
| `index` | int | Row index |
| `scenario` | str | Scenario folder name |
| `stereo left` | str | Relative path to left image |
| `stereo right` | str | Relative path to right image |
| `selected img` | str | (Unused in core pipeline) |
| `label name` | str | Always `true_neg` |
| `label id` | int | Always **−1** (no-gesture class) |
| `roi params` | str | NaN (no gesture ROI for negatives) |
| `synthetic` | int | 0 = real; 1 = synthetic |
| `distortion type` | str | Distortion type (synthetic only) |
| `dist. param 1` | float | Distortion param 1 |
| `dist. param 2` | float | Distortion param 2 |

---

## Stereo Pairing Convention

A left–right stereo pair shares the same **base name** — only the
`_left` / `_right` suffix distinguishes them:

```
{scenario}_{nnnnn}_left.jpg   ←→   {scenario}_{nnnnn}_right.jpg
```

Both are listed explicitly in the CSV (`stereo left` and `stereo right`
columns) so no filename-manipulation heuristic is needed.

---

## Gesture Class Table (Positives)

| label_id | label_name | Gesture Description |
|:---:|---|---|
| 0 | `start_comm` | Start communication |
| 1 | `end_comm` | End communication |
| 2 | `up` | Go up / ascend |
| 3 | `down` | Go down / descend |
| 4 | `photo` | Take photo |
| 5 | `backwards` | Move backwards |
| 6 | `carry` | Carry object |
| 7 | `boat` | Go to boat |
| 8 | `here` | Come here |
| 9 | `mosaic` | Mosaic pattern (survey instruction) |
| 10 | `num_delimiter` | Number delimiter (separates numeric gestures) |
| 11 | `one` | Digit 1 |
| 12 | `two` | Digit 2 |
| 13 | `three` | Digit 3 |
| 14 | `four` | Digit 4 |
| 15 | `five` | Digit 5 |
| −1 | `true_neg` (no-gesture) | No gesture — negatives CSV only |

---

## Scenario ID → Description Mapping

The 8 scenario folders correspond to diving locations and water conditions
as documented in the CADDY paper (Diver Gesture Recognition, Peer et al.).

| Scenario Folder | Location | Water Conditions | Real Positives | Real Negatives |
|---|---|---|---:|---:|
| `biograd-A` | Biograd na Moru, Croatia | Clear, shallow, good visibility | 1,728 | 1,313 |
| `biograd-B` | Biograd na Moru, Croatia | Clear, moderate depth | 975 | 797 |
| `biograd-C` | Biograd na Moru, Croatia | Clear, varied conditions | 1,805 | 1,487 |
| `brodarski-A` | Brodarski Institute pool, Croatia | Controlled pool (clear) | 227 | 220 |
| `brodarski-B` | Brodarski Institute pool, Croatia | Controlled pool (clear) | 87 | 70 |
| `brodarski-C` | Brodarski Institute pool, Croatia | Controlled pool (clear) | 773 | 212 |
| `brodarski-D` | Brodarski Institute pool, Croatia | Controlled pool (clear) | 324 | 304 |
| `genova-A` | Genova, Italy | Open sea, variable turbidity | 3,320 | 2,787 |
| **Total** | | | **9,239** | **7,190** |

> **Note on class imbalance**: `genova-A` (3,320 positives) has ~38× more
> samples than `brodarski-B` (87 positives). This severe scenario imbalance
> directly motivated the Leave-One-Scenario-Out (LOSO) protocol in Phase 2,
> and is also a key factor in stratified splitting decisions (see
> `data/processed/eda_findings.md`).

---

## Locating Any Sample

**To find any labelled sample using only this document:**

1. Identify the `scenario` and whether it is a positive or negative.
2. Navigate to: `CADDY_ROOT/{scenario}/true_positives/raw/` or `.../true_negatives/raw/`
3. The filename format is: `{scenario}_{nnnnn}_left.jpg` (or `_right.jpg`)
4. The exact path is also listed in the relevant CSV's `stereo left` / `stereo right` column.

**Example** — finding gesture "up" in `genova-A`:
```python
import pandas as pd
df = pd.read_csv("data/raw/caddy/CADDY_gestures_complete_v2_release/"
                 "CADDY_gestures_all_true_positives_release_v2.csv")
sample = df[(df['scenario'] == 'genova-A') &
            (df['label name'] == 'up') &
            (df['synthetic'] == 0)].iloc[0]
left_img_path = "data/raw/caddy/CADDY_gestures_complete_v2_release" + sample['stereo left']
```

---

## Source of Truth

The **canonical CSV files** are the two bundled inside
`CADDY_gestures_complete_v2_release.zip`. Any standalone CSVs downloaded
separately must be verified row-count identical before use; if identical,
they are redundant — use only the bundled copies.

Do not maintain two separate CSV files as sources of truth.
