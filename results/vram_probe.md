# P5.8 VRAM Probe Results

**Hardware:** RTX 4060, 8 GB VRAM
**Budget:** < 7.0 GB peak
**Model:** full SENSE (stereo=True, text=True, shared_weights=True), training mode
**Measurement:** forward + backward + AdamW step, 3 timed iters after 2 warm-up, mixed precision (torch.amp.autocast)
**Minimum batch size:** 16 (contrastive loss requirement)

## Probe Table

| image_size | batch_size | peak VRAM (GB) | s / iter | fits (<7.0 GB) | notes |
|---|---|---|---|---|---|
| 380 | 16 | 11.21 | 7.284 | N |  |
| 320 | 16 | 8.16 | 0.938 | N |  |
| 224 | 16 | 4.46 | 0.141 | Y | **SELECTED** |

## Selection Rule

Probe candidates [380, 320, 224] in descending order.
Choose the **largest** size where peak VRAM < 7.0 GB **and** batch_size >= 16.
If two sizes are within 0.1 GB of each other, prefer the faster iteration time.

## Decision

**Selected image_size = 224  batch_size = 16**

Peak VRAM 4.46 GB < 7.0 GB budget. This is the largest candidate that satisfies both the VRAM budget and the minimum-batch-size constraint (>= 16).

## Impact

The chosen values are **frozen** before Phase 7 and must not change after seeing results.
They apply identically to all SENSE variants:
`full`, `no_enhancement`, `waternet`, `mono` (stereo=False), `no_text` (text=False), `neither`.
They feed `train_transforms(image_size)` and `eval_transforms(image_size)`.
The Phase 4 baseline used raw 224x224 (its own config); any resolution difference is a documented limitation (P11.3).
