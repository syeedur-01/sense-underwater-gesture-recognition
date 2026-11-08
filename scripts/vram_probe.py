"""
scripts/vram_probe.py
=====================
P5.8 - VRAM budget check and image-size / batch-size selection.

Probes the heaviest SENSE mode (stereo=True, text=True) in full training mode:
  forward -> throwaway loss -> backward -> AdamW step
using synthetic data (Phase 6 trainer does not exist yet).

Candidate sizes : 224, 320, 380
Minimum batch   : 16  (contrastive loss requires reasonably-sized batches)
Budget          : < 7 GB peak VRAM (RTX 4060, 8 GB total)
Selection rule  : largest size with batch >= 16 that fits; ties broken by iter time.

Output: results/vram_probe.md
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn as nn

# -- make sure project root is on sys.path ---------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.sense import SENSE  # noqa: E402

# --------------------------------------------------------------------------
DEVICE = torch.device("cuda")
NUM_CLASSES = 17
OUT_DIM = 512

# candidates: (image_size, batch_size) pairs to probe in descending size order
CANDIDATES: list[tuple[int, int]] = [
    (380, 16),
    (320, 16),
    (224, 16),
]
VRAM_BUDGET_GB = 7.0
WARMUP_ITERS = 2  # discard to let CUDA settle
TIMED_ITERS = 3  # iterations to average for timing


class ProbeResult(NamedTuple):
    image_size: int
    batch_size: int
    peak_vram_gb: float
    sec_per_iter: float
    fits: bool


def build_model(image_size: int) -> SENSE:
    cfg = {
        "model": {
            "backbone": "efficientnet_b4",
            "pretrained": False,  # no download during probe
            "out_dim": OUT_DIM,
            "num_classes": NUM_CLASSES,
            "shared_weights": True,
            "stereo": {"enabled": True},
            "text_guidance": {
                "enabled": True,
                "encoder_type": "sentence_bert",
                "encoder_name": "all-MiniLM-L6-v2",
                "class_prompt_overrides": {"true_neg": "no gesture"},
            },
            "attention": {"num_heads": 8, "embed_dim": OUT_DIM},
            "classifier": {"temperature": 0.07, "learnable_temperature": True},
        },
        "data": {"image_size": image_size, "num_classes": NUM_CLASSES},
    }
    model = SENSE.from_config(cfg).to(DEVICE)
    model.train()
    return model


def make_batch(image_size: int, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    left = torch.randn(batch_size, 3, image_size, image_size, device=DEVICE)
    right = torch.randn(batch_size, 3, image_size, image_size, device=DEVICE)
    labels = torch.randint(0, NUM_CLASSES, (batch_size,), device=DEVICE)
    return left, right, labels


def probe_one(image_size: int, batch_size: int) -> ProbeResult:
    print(f"\n  Probing size={image_size}  batch={batch_size} ...", flush=True)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(DEVICE)

    model = build_model(image_size)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    def one_iter(
        m: nn.Module,
        opt: torch.optim.Optimizer,
        loss_fn: nn.Module,
    ) -> None:
        left, right, labels = make_batch(image_size, batch_size)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=DEVICE.type):
            logits = m(left, right)
            loss = loss_fn(logits, labels)
        loss.backward()
        opt.step()

    # warm-up
    for _ in range(WARMUP_ITERS):
        one_iter(model, optimizer, criterion)

    torch.cuda.reset_peak_memory_stats(DEVICE)

    # timed
    t0 = time.perf_counter()
    for _ in range(TIMED_ITERS):
        one_iter(model, optimizer, criterion)
    elapsed = (time.perf_counter() - t0) / TIMED_ITERS

    peak_gb = torch.cuda.max_memory_allocated(DEVICE) / 1e9

    # clean up before next probe
    del model, optimizer, criterion
    torch.cuda.empty_cache()

    fits = peak_gb < VRAM_BUDGET_GB
    status = "FITS" if fits else "EXCEEDS"
    print(
        f"    peak VRAM = {peak_gb:.2f} GB  |  {elapsed:.3f} s/iter  |  {status}",
        flush=True,
    )
    return ProbeResult(image_size, batch_size, peak_gb, elapsed, fits)


def select_winner(results: list[ProbeResult]) -> ProbeResult | None:
    """Largest image_size that fits; ties broken by shortest iter time."""
    passing = [r for r in results if r.fits]
    if not passing:
        return None
    passing.sort(key=lambda r: (-r.image_size, r.sec_per_iter))
    return passing[0]


def build_markdown(results: list[ProbeResult], winner: ProbeResult | None) -> str:
    rows = []
    for r in results:
        flag = "**SELECTED**" if (winner and r == winner) else ""
        rows.append(
            f"| {r.image_size} | {r.batch_size} | {r.peak_vram_gb:.2f} |"
            f" {r.sec_per_iter:.3f} | {'Y' if r.fits else 'N'} | {flag} |"
        )
    table = "\n".join(rows)

    if winner:
        decision = (
            f"**Selected image_size = {winner.image_size}  "
            f"batch_size = {winner.batch_size}**\n\n"
            f"Peak VRAM {winner.peak_vram_gb:.2f} GB < {VRAM_BUDGET_GB} GB budget. "
            f"This is the largest candidate that satisfies both the VRAM budget "
            f"and the minimum-batch-size constraint (>= 16)."
        )
    else:
        decision = (
            "**No candidate fit within the VRAM budget at batch_size >= 16.**\n\n"
            "Manual investigation required before proceeding to Phase 7."
        )

    lines = [
        "# P5.8 VRAM Probe Results",
        "",
        "**Hardware:** RTX 4060, 8 GB VRAM",
        f"**Budget:** < {VRAM_BUDGET_GB} GB peak",
        "**Model:** full SENSE (stereo=True, text=True, shared_weights=True)," " training mode",
        f"**Measurement:** forward + backward + AdamW step,"
        f" {TIMED_ITERS} timed iters after {WARMUP_ITERS} warm-up,"
        f" mixed precision (torch.amp.autocast)",
        "**Minimum batch size:** 16 (contrastive loss requirement)",
        "",
        "## Probe Table",
        "",
        "| image_size | batch_size | peak VRAM (GB) | s / iter"
        f" | fits (<{VRAM_BUDGET_GB} GB) | notes |",
        "|---|---|---|---|---|---|",
        table,
        "",
        "## Selection Rule",
        "",
        "Probe candidates [380, 320, 224] in descending order.",
        f"Choose the **largest** size where peak VRAM < {VRAM_BUDGET_GB} GB"
        " **and** batch_size >= 16.",
        "If two sizes are within 0.1 GB of each other, prefer the faster" " iteration time.",
        "",
        "## Decision",
        "",
        decision,
        "",
        "## Impact",
        "",
        "The chosen values are **frozen** before Phase 7 and must not change"
        " after seeing results.",
        "They apply identically to all SENSE variants:",
        "`full`, `no_enhancement`, `waternet`, `mono` (stereo=False),"
        " `no_text` (text=False), `neither`.",
        "They feed `train_transforms(image_size)` and `eval_transforms(image_size)`.",
        "The Phase 4 baseline used raw 224x224 (its own config); any resolution"
        " difference is a documented limitation (P11.3).",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    print("=" * 60)
    print("P5.8  VRAM Probe")
    print(f"Device: {torch.cuda.get_device_name(DEVICE)}")
    total_vram = torch.cuda.get_device_properties(DEVICE).total_memory / 1e9
    print(f"Total VRAM: {total_vram:.2f} GB   Budget: {VRAM_BUDGET_GB} GB")
    print("=" * 60)

    results: list[ProbeResult] = []
    for img_size, bs in CANDIDATES:
        result = probe_one(img_size, bs)
        results.append(result)

    winner = select_winner(results)

    out_path = ROOT / "results" / "vram_probe.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_markdown(results, winner), encoding="utf-8")
    print(f"\nReport written -> {out_path}")

    if winner:
        print(
            f"\n{'='*60}\n"
            f"  SELECTED: image_size={winner.image_size}  "
            f"batch_size={winner.batch_size}  "
            f"({winner.peak_vram_gb:.2f} GB  {winner.sec_per_iter:.3f} s/iter)\n"
            f"{'='*60}"
        )
        print("\nRecord these in configs/sense_full.yaml and configs/base.yaml:")
        print(f"  data.image_size: {winner.image_size}")
        print(f"  training.batch_size: {winner.batch_size}")
    else:
        print("\nNo candidate fitted. See report for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
