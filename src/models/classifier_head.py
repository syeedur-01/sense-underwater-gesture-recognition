"""
src/models/classifier_head.py

P5.5 — CosineSimilarityHead: temperature-scaled cosine similarity classification.

Design rationale (thesis-defensible):
  Instead of a linear (dot-product) head, we classify by computing the cosine
  similarity between the visual feature vector and each class's text embedding.
  This directly implements the geometric interpretation of text guidance: the
  model learns to map visual features into the same embedding space as the
  gesture descriptions, and classifies by proximity.

  The temperature scalar τ scales the similarities before softmax, controlling
  how peaked the distribution is.  A learnable log-parametrized temperature
  (following CLIP, Radford et al. 2021) allows the model to self-calibrate
  this sharpness during training, rather than requiring manual tuning.

  Log-parametrization: τ = exp(log_τ), initialized so τ(0) = 0.07.
  Clamped to [exp(-4.6), exp(4.6)] ≈ [0.01, 100] to prevent gradient
  instability from extremely small or large temperatures.

  Config records: temperature initial value 0.07, learnable=True.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineSimilarityHead(nn.Module):
    """
    Temperature-scaled cosine similarity classification head (P5.5).

    Given a visual feature vector v ∈ R^D and a class embedding matrix
    E ∈ R^(C × D), computes:

        logits = (v / ||v||) · (E / ||E||)ᵀ / τ        ∈ R^(B, C)

    where τ is a learnable, log-parametrized temperature scalar.

    Args:
        embed_dim:   Dimension D of visual feature and class embeddings.
        num_classes: Number of classes C.  Read from config — never hardcoded.
        init_temp:   Initial value of τ (default: 0.07, following CLIP).
        learnable:   If True (default), τ is a learnable parameter.
        temp_min:    Clamp τ to this minimum to prevent collapse.
        temp_max:    Clamp τ to this maximum to prevent instability.
    """

    LOG_TEMP_MIN: float = -4.6  # exp(-4.6) ≈ 0.01
    LOG_TEMP_MAX: float = 4.6  # exp(4.6)  ≈ 100

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        init_temp: float = 0.07,
        learnable: bool = True,
        temp_min: float = 0.01,
        temp_max: float = 100.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.temp_min = temp_min
        self.temp_max = temp_max

        log_init = math.log(init_temp)
        if learnable:
            self.log_temperature = nn.Parameter(torch.tensor(log_init))
        else:
            self.register_buffer("log_temperature", torch.tensor(log_init))

    @property
    def temperature(self) -> torch.Tensor:
        """Current temperature value τ (clamped)."""
        return self.log_temperature.clamp(math.log(self.temp_min), math.log(self.temp_max)).exp()

    def forward(
        self,
        visual: torch.Tensor,
        class_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            visual:           (B, D) fused visual feature.
            class_embeddings: (C, D) or (B, C, D) class text embeddings.

        Returns:
            logits: (B, C) temperature-scaled cosine similarities, always
                computed (and returned) in fp32 regardless of autocast dtype.
        """
        # --- Numerical hardening (audit item 2) --------------------------------
        # The entire head computation is pinned to fp32.  Under autocast, the
        # matmul would otherwise be downcast to fp16 (and autocast's op list
        # can change between PyTorch versions); wrapping in
        # ``autocast(enabled=False)`` with fp32 inputs makes the overflow
        # safety explicit rather than dependent on autocast behavior.
        with torch.autocast(device_type=visual.device.type, enabled=False):
            visual_f32 = visual.float()
            class_embeddings_f32 = class_embeddings.float()

            # L2-normalize visual features: (B, D)
            v_norm = F.normalize(visual_f32, p=2, dim=-1)

            # L2-normalize class embeddings: (C, D)
            if class_embeddings_f32.dim() == 3:
                # (B, C, D) → use the first batch element's embeddings (all same if no proj)
                e_norm = F.normalize(class_embeddings_f32, p=2, dim=-1)  # (B, C, D)
                # Batched matmul: (B, D) · (B, D, C) = (B, C)
                logits = torch.bmm(v_norm.unsqueeze(1), e_norm.transpose(1, 2)).squeeze(1)
            else:
                e_norm = F.normalize(class_embeddings_f32, p=2, dim=-1)  # (C, D)
                # (B, D) · (D, C) = (B, C)
                logits = v_norm @ e_norm.t()

            return logits / self.temperature
