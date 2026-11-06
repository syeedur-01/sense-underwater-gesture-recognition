"""
tests/test_enhancers.py
Phase 3 / P3.3 + P3.4 — Shape, range, and architecture tests for
FUnIEGANEnhancer / WaterNetEnhancer (run without real weight files where possible).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.enhancers import (
    FUnIEGANEnhancer,
    WaterNetEnhancer,
    _FUnIEGANGenerator,
    _pad_to_multiple,
    _to_numpy,
    _to_tensor,
    _unpad,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


class TestUtils:
    """Tests for shared tensor/numpy conversion helpers."""

    def test_to_tensor_shape_dtype(self):
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        t = _to_tensor(img, DEVICE)
        assert t.shape == (1, 3, 240, 320)
        assert t.dtype == torch.float32
        assert t.min().item() >= 0.0 and t.max().item() <= 1.0

    def test_to_numpy_shape_dtype(self):
        t = torch.ones(1, 3, 100, 100)
        arr = _to_numpy(t)
        assert arr.shape == (100, 100, 3)
        assert arr.dtype == np.uint8
        assert arr.max() == 255

    def test_to_numpy_clamp(self):
        t = torch.full((1, 3, 10, 10), 2.0)  # > 1.0 — should clamp
        arr = _to_numpy(t)
        assert arr.max() == 255

    def test_pad_to_multiple_32(self):
        t = torch.zeros(1, 3, 240, 310)
        padded, _ = _pad_to_multiple(t, 32)
        _, _, h, w = padded.shape
        assert h % 32 == 0 and w % 32 == 0
        assert h >= 240 and w >= 310

    def test_pad_to_multiple_already_aligned(self):
        t = torch.zeros(1, 3, 256, 320)
        padded, _ = _pad_to_multiple(t, 32)
        assert padded.shape == t.shape  # no padding needed

    def test_unpad_restores_original_size(self):
        t = torch.zeros(1, 3, 260, 325)
        padded, _ = _pad_to_multiple(t, 32)
        restored = _unpad(padded, 260, 325)
        assert restored.shape == (1, 3, 260, 325)


# ---------------------------------------------------------------------------
# FUnIE-GAN Generator (without weights)
# ---------------------------------------------------------------------------


class TestFUnIEGANGenerator:
    """Shape and range tests using random weights (no .pth file needed)."""

    @pytest.fixture(scope="class")
    @classmethod
    def model(cls):
        return _FUnIEGANGenerator().to(DEVICE).eval()

    def test_output_shape_exact_multiple(self, model):
        """Generator output is (B, 3, H-2, W-2) for a mult-of-32 input."""
        x = torch.randn(1, 3, 256, 256).to(DEVICE)
        with torch.no_grad():
            out = model(x)
        assert out.shape[1] == 3
        # The final ZeroPad(1,0,1,0)+Conv(k=4,p=0) shrinks H and W by 2 each
        assert out.shape[2] == 254
        assert out.shape[3] == 254

    def test_output_range_tanh(self, model):
        x = torch.randn(1, 3, 128, 128).to(DEVICE)
        with torch.no_grad():
            out = model(x)
        assert out.min().item() >= -1.0 - 1e-3
        assert out.max().item() <= 1.0 + 1e-3

    def test_full_pipeline_preserves_spatial_size(self, model):
        """Full pipeline via FUnIEGANEnhancer.enhance() preserves spatial size."""
        import math

        dummy = np.random.randint(10, 240, (240, 320, 3), dtype=np.uint8)
        orig_h, orig_w = dummy.shape[:2]
        x = _to_tensor(dummy, DEVICE)
        # Use the same padding strategy as FUnIEGANEnhancer.enhance():
        # pad to ceil((orig+2)/32)*32 so that model output (H-2) >= orig
        import torch.nn.functional as F_

        target_h = math.ceil((orig_h + 2) / 32) * 32
        target_w = math.ceil((orig_w + 2) / 32) * 32
        pad_h, pad_w = target_h - orig_h, target_w - orig_w
        pad_mode = "reflect" if (pad_h < orig_h and pad_w < orig_w) else "replicate"
        x_pad = F_.pad(x, (0, pad_w, 0, pad_h), mode=pad_mode)
        with torch.no_grad():
            out = model(x_pad)
        out_norm = (out + 1.0) / 2.0
        out_crop = _unpad(out_norm, orig_h, orig_w)
        result = _to_numpy(out_crop)
        assert result.shape == dummy.shape
        assert result.dtype == np.uint8

    def test_various_input_sizes(self, model):
        """Various sizes via the FUnIEGANEnhancer-compatible pipeline."""
        import math

        import torch.nn.functional as F_

        sizes = [(100, 100), (192, 256), (480, 640)]
        for h, w in sizes:
            dummy = np.random.randint(10, 240, (h, w, 3), dtype=np.uint8)
            x = _to_tensor(dummy, DEVICE)
            target_h = math.ceil((h + 2) / 32) * 32
            target_w = math.ceil((w + 2) / 32) * 32
            pad_h, pad_w = target_h - h, target_w - w
            pad_mode = "reflect" if (pad_h < h and pad_w < w) else "replicate"
            x_pad = F_.pad(x, (0, pad_w, 0, pad_h), mode=pad_mode)
            with torch.no_grad():
                out = model(x_pad)
            out_crop = _unpad((out + 1.0) / 2.0, h, w)
            result = _to_numpy(out_crop)
            assert result.shape == (h, w, 3), f"Failed for size ({h}, {w})"


# ---------------------------------------------------------------------------
# FUnIEGANEnhancer — missing weights error
# ---------------------------------------------------------------------------


class TestEnhancerWeightErrors:
    def test_funiegan_missing_weights_raises(self, tmp_path):
        fake_path = tmp_path / "nonexistent.pth"
        with pytest.raises(FileNotFoundError, match="FUnIE-GAN weights not found"):
            FUnIEGANEnhancer(weights_path=fake_path)


# ---------------------------------------------------------------------------
# WaterNetEnhancer — torch.hub integration (loads real weights from internet)
# ---------------------------------------------------------------------------


class TestWaterNetEnhancer:
    """
    Integration test for WaterNetEnhancer via torch.hub.
    Downloads pretrained weights on first run (~10 MB, cached afterwards).
    Mark with --run-hub-tests to execute (skipped by default to keep CI fast).
    """

    @pytest.fixture(scope="class")
    @classmethod
    def enhancer(cls):
        return WaterNetEnhancer(device=str(DEVICE))

    def test_enhance_shape_and_dtype(self, enhancer):
        dummy = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)
        result = enhancer.enhance(dummy)
        assert result.shape == dummy.shape, f"Shape mismatch: {result.shape}"
        assert result.dtype == np.uint8

    def test_enhance_various_sizes(self, enhancer):
        # Use random non-trivial images (all-zeros causes white-balance div-by-zero in WaterNet)
        rng = np.random.default_rng(42)
        for h, w in [(100, 100), (192, 256), (480, 640)]:
            dummy = rng.integers(10, 256, (h, w, 3), dtype=np.uint8)
            result = enhancer.enhance(dummy)
            assert result.shape == (h, w, 3), f"Failed for size ({h}, {w})"

    def test_enhance_output_range(self, enhancer):
        dummy = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        result = enhancer.enhance(dummy)
        assert result.min() >= 0 and result.max() <= 255
