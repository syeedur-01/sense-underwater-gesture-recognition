"""
src/models/sense.py

P5.6 — Full SENSE model assembly.
P5.7 — Ablation-mode switches (stereo.enabled, text_guidance.enabled) built as
        config-driven branches inside the SAME class — required for Phase 10's
        ablations to be fair, apples-to-apples comparisons.

SENSE — Stereo-Enhanced kNowledge-guided Gesture Encoder
=========================================================
Architecture (full mode):
  BiChannelEncoder (EfficientNet-B4 × 2, shared weights)
    → SelfAttentionBlock
    → CrossModalFusion ← GestureTextEncoder
    → CosineSimilarityHead (learnable temperature, τ₀=0.07)

Ablation modes (config-driven, same class):
  stereo_enabled=False : BiChannelEncoder runs left branch only → (B, D)
  text_enabled=False   : CrossModalFusion + cosine head replaced by a
                         plain linear head — same parameter count path as
                         much as possible so comparisons are fair.
  Both False           : mono + linear head (simplest ablation).

The class reads all structural hyperparameters from a flat config dict
(produced by load_yaml_config) so the architecture is fully determined
by the YAML file, not by magic constants in this file.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.attention import SelfAttentionBlock
from src.models.classifier_head import CosineSimilarityHead
from src.models.fusion import CrossModalFusion
from src.models.text_encoder import CADDY_CLASS_NAMES, GestureTextEncoder
from src.models.visual_encoder import BiChannelEncoder


class SENSE(nn.Module):
    """
    Full SENSE model with ablation switches (P5.6 / P5.7).

    Args:
        num_classes:      Number of gesture classes.  Read from config (never 17-hardcoded here).
        backbone:         timm model name for the visual backbone.
        pretrained:       Whether to load ImageNet pretrained weights.
        out_dim:          Feature dimension per branch after linear projection.
        shared_weights:   Whether BiChannelEncoder branches share weights.
        stereo_enabled:   True → use both left+right images; False → left only.
        text_enabled:     True → CrossModalFusion + cosine head;
                          False → plain linear head (ablation).
        num_heads:        Number of attention heads for SelfAttentionBlock and
                          CrossModalFusion.
        attn_dropout:     Dropout probability on attention outputs.
        encoder_type:     Text backbone type (passed to GestureTextEncoder).
        encoder_name:     Specific model name for the text backbone.
        class_names:      Class names ordered by gesture_id.
        class_prompt_overrides: Prompt overrides (e.g. {'true_neg': 'no gesture'}).
        init_temp:        Initial cosine-head temperature τ₀.
        learnable_temp:   Whether τ is a learnable parameter.
    """

    def __init__(
        self,
        num_classes: int,
        backbone: str = "efficientnet_b4",
        pretrained: bool = True,
        out_dim: int = 512,
        shared_weights: bool = True,
        stereo_enabled: bool = True,
        text_enabled: bool = True,
        num_heads: int = 8,
        attn_dropout: float = 0.1,
        encoder_type: str = "sentence_bert",
        encoder_name: str = "all-MiniLM-L6-v2",
        class_names: list[str] | None = None,
        class_prompt_overrides: dict[str, str] | None = None,
        init_temp: float = 0.07,
        learnable_temp: bool = True,
    ) -> None:
        super().__init__()

        self.stereo_enabled = stereo_enabled
        self.text_enabled = text_enabled
        self.num_classes = num_classes

        if class_names is None:
            class_names = CADDY_CLASS_NAMES
        if class_prompt_overrides is None:
            class_prompt_overrides = {"true_neg": "no gesture"}

        # --- P5.1: Visual encoder ---
        self.encoder = BiChannelEncoder(
            backbone=backbone,
            pretrained=pretrained,
            out_dim=out_dim,
            shared_weights=shared_weights,
            stereo_enabled=stereo_enabled,
        )
        visual_dim = self.encoder.feature_dim  # out_dim if mono, 2*out_dim if stereo

        # --- P5.2: Self-attention refinement ---
        self.self_attn = SelfAttentionBlock(
            embed_dim=visual_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
        )

        if text_enabled:
            # --- P5.3: Text encoder ---
            # Project text embeddings to match visual_dim
            self.text_encoder = GestureTextEncoder(
                class_names=class_names,
                overrides=class_prompt_overrides,
                encoder_type=encoder_type,
                encoder_name=encoder_name,
                out_dim=visual_dim,
            )

            # --- P5.4: Cross-modal fusion ---
            self.fusion = CrossModalFusion(
                embed_dim=visual_dim,
                num_heads=num_heads,
                dropout=attn_dropout,
            )

            # --- P5.5: Cosine similarity head ---
            self.head = CosineSimilarityHead(
                embed_dim=visual_dim,
                num_classes=num_classes,
                init_temp=init_temp,
                learnable=learnable_temp,
            )

        else:
            # Ablation: no text guidance → plain linear classifier
            # (CrossModalFusion and GestureTextEncoder are not instantiated)
            self.text_encoder = None
            self.fusion = None
            self.head = nn.Linear(visual_dim, num_classes)

    @classmethod
    def from_config(cls, cfg: dict) -> "SENSE":
        """
        Construct SENSE from a loaded YAML config dict.

        Expected keys (all under top-level sections matching base.yaml):
          model.num_classes, model.backbone, model.pretrained, model.out_dim,
          model.shared_weights, model.stereo.enabled,
          model.text_guidance.enabled, model.text_guidance.encoder_type,
          model.text_guidance.encoder_name,
          model.text_guidance.class_prompt_overrides,
          model.attention.num_heads, model.classifier.temperature,
          data.num_classes (fallback if model.num_classes absent).
        """
        m = cfg.get("model", {})
        tg = m.get("text_guidance", {})
        attn = m.get("attention", {})
        cls_cfg = m.get("classifier", {})
        data = cfg.get("data", {})

        num_classes = m.get("num_classes") or data.get("num_classes", 17)

        return cls(
            num_classes=num_classes,
            backbone=m.get("backbone", "efficientnet_b4"),
            pretrained=m.get("pretrained", True),
            out_dim=m.get("out_dim", 512),
            shared_weights=m.get("shared_weights", True),
            stereo_enabled=m.get("stereo", {}).get("enabled", True),
            text_enabled=tg.get("enabled", True),
            num_heads=attn.get("num_heads", 8),
            attn_dropout=attn.get("dropout", 0.1),
            encoder_type=tg.get("encoder_type", "sentence_bert"),
            encoder_name=tg.get("encoder_name", "all-MiniLM-L6-v2"),
            class_prompt_overrides=tg.get("class_prompt_overrides", {"true_neg": "no gesture"}),
            init_temp=cls_cfg.get("temperature", 0.07),
            learnable_temp=cls_cfg.get("learnable_temperature", True),
        )

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            left:  (B, 3, H, W) left image.
            right: (B, 3, H, W) right image (required when stereo_enabled=True).

        Returns:
            logits: (B, num_classes).
        """
        # 1. Visual encoding
        visual = self.encoder(left, right)  # (B, D) or (B, 2D)

        # 2. Self-attention refinement
        visual = self.self_attn(visual)  # (B, D) or (B, 2D)

        # 3. Cross-modal fusion (or linear head in ablation)
        if self.text_enabled:
            text_emb = self.text_encoder()  # (num_classes, D)
            visual = self.fusion(visual, text_emb)  # (B, D)
            logits = self.head(visual, text_emb)  # (B, num_classes)
        else:
            logits = self.head(visual)  # (B, num_classes)

        return logits
