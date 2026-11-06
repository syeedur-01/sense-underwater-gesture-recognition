"""
src/models/enhancers.py
Inference wrappers for FUnIE-GAN and WaterNet underwater image enhancers (P3.3 / P3.4).

Both models are **monocular**: each image (left or right) is enhanced independently.
They are used exclusively for offline cache-precomputation (scripts/cache_enhanced_images.py)
and must NEVER be called inside a training loop.

Architecture references:
  FUnIE-GAN: Islam et al., IEEE RA-L 2020  — https://github.com/xahidbuffon/FUnIE-GAN
  WaterNet:  Li et al.,   IEEE TIP 2019    — PyTorch port: https://github.com/tnwei/waternet
             (original TF repo: https://github.com/Li-Chongyi/Water-Net_Code — not used)

FUnIE-GAN weights: must be downloaded manually to weights/funie_gan/generator.pth
                   See weights/funie_gan/SOURCE.md for instructions.
WaterNet weights:  downloaded automatically via torch.hub on first use.
                   No manual step required. See weights/waternet/SOURCE.md.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import Paths

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def _to_tensor(image_rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Convert an HxWx3 uint8 RGB numpy array to a (1, 3, H, W) float32 tensor
    with values in [0, 1].
    """
    t = torch.from_numpy(image_rgb.astype(np.float32) / 255.0)  # H W C
    return t.permute(2, 0, 1).unsqueeze(0).to(device)  # 1 C H W


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a (1, 3, H, W) float32 tensor with values in [0, 1] to HxWx3 uint8 RGB.
    Clamps and rounds.
    """
    arr = tensor.squeeze(0).permute(1, 2, 0).cpu().float()
    arr = (arr.clamp(0.0, 1.0) * 255.0).round().numpy().astype(np.uint8)
    return arr


def _pad_to_multiple(
    x: torch.Tensor, multiple: int
) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    """
    Pad a (B, C, H, W) tensor so H and W are multiples of `multiple`.
    Returns (padded_tensor, (pad_left, pad_right, pad_top, pad_bottom)).
    """
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    # F.pad order: (left, right, top, bottom)
    padding = (0, pad_w, 0, pad_h)
    return F.pad(x, padding, mode="reflect"), (0, pad_w, 0, pad_h)


def _unpad(x: torch.Tensor, orig_h: int, orig_w: int) -> torch.Tensor:
    """Remove padding — just crop back to original spatial size."""
    return x[:, :, :orig_h, :orig_w]


# ---------------------------------------------------------------------------
# FUnIE-GAN Generator  (Islam et al., 2020) — FUnIE-GAN-UP variant
# ---------------------------------------------------------------------------
# Architecture: 5-block U-Net encoder–decoder matching the released checkpoint.
#   Encoder (down1..5): Conv(k=4,s=2) + optional BatchNorm + LeakyReLU(0.2)
#     down1: 3→32,  no BN
#     down2: 32→128, BN
#     down3: 128→256, BN
#     down4: 256→256, BN
#     down5: 256→256, no BN  (bottleneck)
#   Decoder (up1..4): ConvTranspose(k=4,s=2) + BatchNorm + ReLU + skip concat
#     up1: 256→256   (in=down5=256)
#     up2: 512→256   (in=cat(up1,down4)=512)
#     up3: 512→128   (in=cat(up2,down3)=512)
#     up4: 256→32    (in=cat(up3,down2)=256)
#   Final: Upsample(2×) → ZeroPad2d → Conv(64→3,k=4) → Tanh
#     The final Sequential has Conv at index 2: [Upsample, Pad, Conv]
#     skip concat with down1(32ch) is inserted before Conv in forward().
# Output is in [-1, 1]; post-process → [0, 1].


class _DownBlock(nn.Module):
    """FUnIE-GAN-UP encoder block: Conv(k=4,s=2,p=1) + optional BatchNorm + LeakyReLU."""

    def __init__(self, in_ch: int, out_ch: int, normalize: bool = True):
        super().__init__()
        # bias=False in all cases: when BN is present it absorbs bias;
        # when normalize=False the checkpoint also has no bias term.
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False)
        ]
        if normalize:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class _UpBlock(nn.Module):
    """FUnIE-GAN-UP decoder block: ConvTranspose(k=4,s=2,p=1) + BatchNorm + ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class _FUnIEGANGenerator(nn.Module):
    """
    FUnIE-GAN-UP Generator.

    Matches the released checkpoint (46 keys):
      down1..5, up1..4, final.2 (Conv2d 64→3, kernel 4×4).

    Input:  (B, 3, H, W)  — H and W must be multiples of 32.
    Output: (B, 3, H, W)  — values in [-1, 1] (tanh).
    """

    def __init__(self):
        super().__init__()
        # Encoder
        self.down1 = _DownBlock(3, 32, normalize=False)  # /2  → 32ch
        self.down2 = _DownBlock(32, 128)  # /4  → 128ch
        self.down3 = _DownBlock(128, 256)  # /8  → 256ch
        self.down4 = _DownBlock(256, 256)  # /16 → 256ch
        self.down5 = _DownBlock(256, 256, normalize=False)  # /32 → 256ch (bottleneck)

        # Decoder with skip connections
        self.up1 = _UpBlock(256, 256)  # in: down5(256)              → 256ch  /16
        self.up2 = _UpBlock(256 + 256, 256)  # in: cat(up1,down4) = 512ch  → 256ch  /8
        self.up3 = _UpBlock(256 + 256, 128)  # in: cat(up2,down3) = 512ch  → 128ch  /4
        self.up4 = _UpBlock(128 + 128, 32)  # in: cat(up3,down2) = 256ch  → 32ch   /2

        # Final: Upsample up4(32ch) → concat with down1(32ch) = 64ch → Conv(64→3)
        # State-dict has keys final.0 (no params), final.1 (no params), final.2 (Conv weights).
        # We split the sequential so the skip-concat can happen between index 1 and index 2.
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),  # index 0 — no params
            nn.ZeroPad2d((1, 0, 1, 0)),  # index 1 — no params
            nn.Conv2d(64, 3, kernel_size=4, stride=1, padding=0),  # index 2 — has weights
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        # Decode with skip connections
        u1 = self.up1(d5)
        u2 = self.up2(torch.cat([u1, d4], dim=1))
        u3 = self.up3(torch.cat([u2, d3], dim=1))
        u4 = self.up4(torch.cat([u3, d2], dim=1))

        # Final: concat u4(H/2,32ch) with d1(H/2,32ch) → 64ch
        # Then: Upsample(2×) → ZeroPad → Conv(64→3) → Tanh
        merged = torch.cat([u4, d1], dim=1)  # 32 + 32 = 64ch at H/2
        return torch.tanh(self.final(merged))  # final: Upsample→Pad→Conv


class FUnIEGANEnhancer:
    """
    Inference wrapper for the FUnIE-GAN generator (P3.3).

    Usage::

        enhancer = FUnIEGANEnhancer(device="cuda")
        enhanced_rgb = enhancer.enhance(raw_rgb_uint8)   # np.ndarray HxWx3 uint8

    The weight file must exist at ``Paths.FUNIE_GAN_WEIGHTS``
    (default: ``weights/funie_gan/generator.pth``).
    """

    def __init__(
        self,
        weights_path: Union[str, Path, None] = None,
        device: Union[str, torch.device] = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        weights_path = Path(weights_path or Paths.FUNIE_GAN_WEIGHTS)

        if not weights_path.exists():
            raise FileNotFoundError(
                f"FUnIE-GAN weights not found at: {weights_path}\n"
                f"See weights/funie_gan/SOURCE.md for download instructions."
            )

        self.model = _FUnIEGANGenerator().to(self.device)
        state = torch.load(weights_path, map_location=self.device, weights_only=False)
        # Accept bare state_dict or wrapped checkpoint
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def enhance(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Enhance a single RGB image.

        Args:
            image_rgb: HxWx3 uint8 numpy array (RGB).

        Returns:
            Enhanced HxWx3 uint8 numpy array (RGB), same spatial size as input.
        """
        orig_h, orig_w = image_rgb.shape[:2]
        x = _to_tensor(image_rgb, self.device)
        # The generator's final ZeroPad2d(1,0,1,0)+Conv(k=4,p=0) reduces each spatial
        # dimension by 2 pixels (H_out = H_in - 2).  To preserve size we need to pad the
        # input so that (padded_H - 2) >= orig_H, i.e. padded_H >= orig_H + 2, AND
        # padded_H must be a multiple of 32 for skip-connection alignment.
        target_h = math.ceil((orig_h + 2) / 32) * 32  # smallest mult-of-32 >= orig_h+2
        target_w = math.ceil((orig_w + 2) / 32) * 32
        pad_h = target_h - orig_h
        pad_w = target_w - orig_w
        # reflect requires pad < dim; fall back to replicate for very small inputs
        pad_mode = "reflect" if (pad_h < orig_h and pad_w < orig_w) else "replicate"
        x = F.pad(x, (0, pad_w, 0, pad_h), mode=pad_mode)
        out = self.model(x)  # tanh → [-1, 1], H_out = target_h - 2
        out = (out + 1.0) / 2.0  # → [0, 1]
        out = _unpad(out, orig_h, orig_w)  # crop to original size
        return _to_numpy(out)


# ---------------------------------------------------------------------------
# WaterNet  (Li et al., 2019) — via tnwei/waternet PyTorch port
# ---------------------------------------------------------------------------
# Weights are auto-downloaded by torch.hub on first instantiation.
# No manual .pth file is required.
#
# Hub entrypoints used:
#   preprocess = torch.hub.load('tnwei/waternet', 'arr2ten_noeinops')
#   postprocess = torch.hub.load('tnwei/waternet', 'ten2arr_noeinops')
#   model      = torch.hub.load('tnwei/waternet', 'waternet')
#
# Input  : RGB HxWx3 uint8 numpy array
# Output : Enhanced RGB HxWx3 uint8 numpy array (same spatial size)


class WaterNetEnhancer:
    """
    Inference wrapper for WaterNet (P3.4).

    Uses the community PyTorch port ``tnwei/waternet`` loaded via ``torch.hub``.
    Pretrained weights are downloaded automatically on first use
    (cached to ``~/.cache/torch/hub/checkpoints/``).

    **No manual weight file download required.**
    See ``weights/waternet/SOURCE.md`` for the paper citation.

    Usage::

        enhancer = WaterNetEnhancer(device="cuda")
        enhanced_rgb = enhancer.enhance(raw_rgb_uint8)   # np.ndarray HxWx3 uint8
    """

    _HUB_REPO = "tnwei/waternet"

    def __init__(
        self,
        device: Union[str, torch.device] = "cuda",
        trust_repo: bool = True,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # torch.hub.load('tnwei/waternet', 'waternet') returns a 3-tuple:
        #   (preprocess_fn, postprocess_fn, model)
        # Pretrained weights are downloaded automatically on first call
        # and cached to ~/.cache/torch/hub/checkpoints/.
        result = torch.hub.load(self._HUB_REPO, "waternet", trust_repo=trust_repo, verbose=False)
        self._preprocess, self._postprocess, self.model = result
        self.model.to(self.device).eval()

    @torch.no_grad()
    def enhance(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Enhance a single RGB image.

        Args:
            image_rgb: HxWx3 uint8 numpy array (RGB).

        Returns:
            Enhanced HxWx3 uint8 numpy array (RGB), same spatial size as input.
        """
        # preprocess returns (rgb_ten, wb_ten, he_ten, gc_ten) on CPU
        rgb_ten, wb_ten, he_ten, gc_ten = self._preprocess(image_rgb)

        # Move all tensors to the target device
        rgb_ten = rgb_ten.to(self.device)
        wb_ten = wb_ten.to(self.device)
        he_ten = he_ten.to(self.device)
        gc_ten = gc_ten.to(self.device)

        out_ten = self.model(rgb_ten, wb_ten, he_ten, gc_ten)

        # postprocess returns a numpy array; squeeze batch dim if present (B, H, W, 3) → (H, W, 3)
        out = self._postprocess(out_ten)
        if out.ndim == 4:
            out = out.squeeze(0)
        return out.astype(np.uint8)
