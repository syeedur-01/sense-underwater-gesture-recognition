"""
src/utils/seed.py
=================
P6.7 — Deterministic seed control across Python / NumPy / PyTorch / CUDA.

Usage:
    from src.utils.seed import set_seed
    set_seed(42)

Calling ``set_seed`` sets:
  - Python built-in ``random`` module
  - NumPy random state
  - PyTorch CPU + CUDA random state
  - ``torch.backends.cudnn.deterministic = True``
  - ``torch.backends.cudnn.benchmark = False``

Trade-offs:
  - ``cudnn.deterministic=True`` disables cuDNN auto-tuning (small throughput
    cost) in exchange for bit-reproducible results.
  - ``cudnn.benchmark=False`` prevents input-shape-dependent algorithm
    selection, which would vary between runs.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """
    Seed all random-number generators for reproducibility.

    Args:
        seed: Integer seed value.  The same seed on the same hardware
              will produce identical epoch-1 loss values.
    """
    seed = int(seed)

    # Python built-in
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch (CPU + CUDA)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # multi-GPU

    # cuDNN determinism
    cudnn.deterministic = True
    cudnn.benchmark = False

    # PYTHONHASHSEED for dict ordering etc.
    os.environ["PYTHONHASHSEED"] = str(seed)

    logger.info("Global seed set to %d (cudnn.deterministic=True, benchmark=False).", seed)
