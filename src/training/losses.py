"""
src/training/losses.py
======================
P6.1 — Combined loss function for SENSE.

Loss formulation:
    SenseLoss = alpha * CrossEntropy + beta * ContrastiveLoss

Key specifications:
    - CrossEntropy accepts an optional class weight tensor from
      compute_class_weights() (P2.9), applied when training.loss.class_weighted: true
      (default true for all SENSE runs).
    - When training.loss.class_weighted: false (e.g. Phase 4 baseline), plain
      unweighted CrossEntropy is used for literature comparability.
    - ContrastiveLoss implements an InfoNCE-style bidirectional image-text
      contrastive term across the batch.
    - Gracefully degrades to CrossEntropy-only when text_guidance.enabled: false
      (or beta == 0.0), independent of the class-weighting setting.
    - Outputs a positive, finite scalar supporting backward().
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.splits import compute_class_weights


class ContrastiveLoss(nn.Module):
    """
    InfoNCE-style bidirectional image-text contrastive loss (P6.1).

    Given a batch of image representations and paired text descriptions,
    contrasts positive pairs against in-batch negative pairs in both
    directions (image-to-text and text-to-image).

    Can operate directly on temperature-scaled cosine similarity logits
    of shape (B, num_classes) or on explicit feature representations.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        logits_or_features: torch.Tensor,
        targets: torch.Tensor,
        text_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            logits_or_features: Either (B, C) similarity logits from CosineSimilarityHead,
                                or (B, D) visual feature vectors.
            targets:            (B,) ground truth class indices.
            text_embeddings:    Optional (C, D) or (B, D) class text embeddings.
                                If None, logits_or_features is treated as (B, C) logits.

        Returns:
            Scalar contrastive loss tensor.
        """
        device = logits_or_features.device
        batch_size = logits_or_features.shape[0]

        if batch_size <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # --- Numerical hardening (audit item 2) --------------------------------
        # The similarity computation and both InfoNCE cross-entropies always
        # run in fp32.  Under autocast, the matmul would otherwise be downcast
        # to fp16; wrapping in ``autocast(enabled=False)`` with fp32 inputs
        # keeps this region overflow-safe independent of the autocast op list
        # (see CosineSimilarityHead.forward for rationale).
        with torch.autocast(device_type=device.type, enabled=False):
            sim_input = logits_or_features.float()

            if text_embeddings is not None:
                # Explicit feature vectors provided: compute cosine similarities
                v_norm = F.normalize(sim_input, p=2, dim=-1)
                text_embeddings = text_embeddings.float()
                if text_embeddings.dim() == 2 and text_embeddings.shape[0] != batch_size:
                    # (C, D) class embeddings: project to (B, C) similarity
                    t_norm = F.normalize(text_embeddings, p=2, dim=-1)
                    sim_matrix = (v_norm @ t_norm.t()) / self.temperature  # (B, C)
                    # In-batch cross-modal similarity matrix (B, B)
                    batch_sim = sim_matrix[:, targets]
                else:
                    # (B, D) already-paired text embeddings
                    t_norm = F.normalize(text_embeddings, p=2, dim=-1)
                    batch_sim = (v_norm @ t_norm.t()) / self.temperature  # (B, B)
            else:
                # (B, C) logits already contain temperature-scaled cosine similarities
                # Sub-sample class similarities corresponding to batch targets -> (B, B)
                batch_sim = sim_input[:, targets]

            labels = torch.arange(batch_size, device=device)

            # Bidirectional InfoNCE: Image-to-Text and Text-to-Image
            loss_i2t = F.cross_entropy(batch_sim, labels)
            loss_t2i = F.cross_entropy(batch_sim.t(), labels)

            return 0.5 * (loss_i2t + loss_t2i)


class SenseLoss(nn.Module):
    """
    Combined loss function for SENSE:
        loss = alpha * CrossEntropy + beta * ContrastiveLoss

    Args:
        alpha:          Weight for classification loss (default: 0.7).
        beta:           Weight for contrastive loss (default: 0.3).
        class_weights:  Optional tensor of inverse-frequency class weights
                        from compute_class_weights() of shape (num_classes,).
        text_enabled:   If False, beta is set to 0 and loss is CE-only.
        temperature:    Temperature for contrastive loss (default: 0.07).
    """

    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        class_weights: torch.Tensor | None = None,
        text_enabled: bool = True,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.text_enabled = text_enabled
        self.temperature = temperature

        if not text_enabled or beta <= 0.0:
            self.alpha = 1.0
            self.beta = 0.0
        else:
            self.alpha = alpha
            self.beta = beta

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.clone().detach().float())
            self.ce = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.class_weights = None
            self.ce = nn.CrossEntropyLoss()

        if self.beta > 0.0:
            self.contrastive = ContrastiveLoss(temperature=temperature)
        else:
            self.contrastive = None

        # Tracking variables for recent batch
        self.last_total_loss: float = 0.0
        self.last_ce_loss: float = 0.0
        self.last_contrastive_loss: float = 0.0

    @classmethod
    def from_config(
        cls,
        cfg: dict[str, Any],
        train_csv: str | Path | None = None,
        device: torch.device | str | None = None,
    ) -> "SenseLoss":
        """
        Build SenseLoss from configuration dictionary.

        Resolves alpha, beta, and class weights computed fresh from train_csv
        when training.loss.class_weighted: true.
        """
        train_cfg = cfg.get("training", {})
        loss_cfg = train_cfg.get("loss", {})
        model_cfg = cfg.get("model", {})
        data_cfg = cfg.get("data", {})

        alpha = float(loss_cfg.get("alpha", 0.7))
        beta = float(loss_cfg.get("beta", 0.3))
        class_weighted = bool(loss_cfg.get("class_weighted", True))
        text_enabled = bool(model_cfg.get("text_guidance", {}).get("enabled", True))
        num_classes = int(model_cfg.get("num_classes") or data_cfg.get("num_classes", 17))
        temp = float(model_cfg.get("classifier", {}).get("temperature", 0.07))

        class_weights = None
        if class_weighted:
            resolved_csv = train_csv or data_cfg.get("split_train")
            if resolved_csv is not None and Path(resolved_csv).exists():
                class_weights = compute_class_weights(resolved_csv, num_classes=num_classes)
            else:
                # If path not found or not given, warn or compute uniform
                pass

        loss_fn = cls(
            alpha=alpha,
            beta=beta,
            class_weights=class_weights,
            text_enabled=text_enabled,
            temperature=temp,
        )

        if device is not None:
            loss_fn = loss_fn.to(torch.device(device))

        return loss_fn

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        text_embeddings: torch.Tensor | None = None,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """
        Forward pass for combined loss.

        Args:
            logits:            (B, num_classes) predictions.
            targets:           (B,) target class indices.
            text_embeddings:   Optional text embeddings if explicit projection used.
            return_components: If True, returns (total_loss, metrics_dict).

        Returns:
            total_loss (scalar tensor) or (total_loss, metrics_dict).
        """
        # Numerical hardening (audit item 2): CE is always computed on fp32
        # logits.  The cosine head already returns fp32, but the text-disabled
        # ablation arms use a plain nn.Linear head that emits fp16 under autocast.
        ce_loss = self.ce(logits.float(), targets)

        if self.text_enabled and self.beta > 0.0 and self.contrastive is not None:
            contrastive_loss = self.contrastive(logits, targets, text_embeddings=text_embeddings)
            total_loss = self.alpha * ce_loss + self.beta * contrastive_loss
        else:
            contrastive_loss = torch.tensor(0.0, device=logits.device)
            total_loss = ce_loss

        self.last_total_loss = float(total_loss.detach().item())
        self.last_ce_loss = float(ce_loss.detach().item())
        self.last_contrastive_loss = float(contrastive_loss.detach().item())

        if return_components:
            components = {
                "loss": self.last_total_loss,
                "ce_loss": self.last_ce_loss,
                "contrastive_loss": self.last_contrastive_loss,
            }
            return total_loss, components

        return total_loss

    def get_last_metrics(self) -> dict[str, float]:
        """Return the dictionary of metrics from the most recent forward pass."""
        return {
            "loss": self.last_total_loss,
            "ce_loss": self.last_ce_loss,
            "contrastive_loss": self.last_contrastive_loss,
        }
