"""
src/models/fusion.py

P5.4 — CrossModalFusion: bidirectional cross-attention between visual features
        and text embeddings.

Design rationale (thesis-defensible):
  Given a visual feature vector v ∈ R^D (from SelfAttentionBlock) and a
  class embedding matrix T ∈ R^(C×D) (from GestureTextEncoder), we want the
  visual features to attend to class descriptions *and* class descriptions to
  attend to the visual features.  This bidirectional design lets the model both:
    (1) identify which gesture-text description best matches the current visual
        input (visual→text direction), and
    (2) refine the visual representation by injecting semantic context from the
        most-attended class embedding (text→visual direction).

  The two cross-attention sub-layers each follow Pre-LN (LayerNorm before
  attention) for training stability.  Outputs of both directions are
  concatenated and projected back to D via a linear layer, giving a fused
  representation of the same dimensionality as the input visual feature.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossModalFusion(nn.Module):
    """
    Bidirectional cross-attention fusion (P5.4).

    Takes:
      - ``visual``: (B, D) — visual feature from SelfAttentionBlock.
      - ``text_emb``: (C, D) or (B, C, D) — class text embeddings from
        GestureTextEncoder.  If (C, D), it is broadcast over the batch.

    Returns:
      - Fused visual feature (B, D).

    Args:
        embed_dim: Feature dimension D.  Must match visual input and text_emb dim.
        num_heads: Number of attention heads.  Must divide embed_dim evenly.
        dropout:   Dropout on attention outputs.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # Visual → Text: query=visual, key/value=text
        self.norm_v1 = nn.LayerNorm(embed_dim)
        self.norm_t1 = nn.LayerNorm(embed_dim)
        self.v2t_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Text → Visual: query=text, key/value=visual
        self.norm_t2 = nn.LayerNorm(embed_dim)
        self.norm_v2 = nn.LayerNorm(embed_dim)
        self.t2v_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)

        # Project concatenated bidirectional output back to embed_dim
        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        visual: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            visual:   (B, D) visual feature vector.
            text_emb: (C, D) or (B, C, D) class text embeddings.

        Returns:
            Fused feature tensor (B, D).
        """
        B = visual.shape[0]

        # Expand visual to sequence: (B, 1, D)
        v_seq = visual.unsqueeze(1)

        # Expand text to (B, C, D) if needed
        if text_emb.dim() == 2:  # (C, D) → (B, C, D)
            t_seq = text_emb.unsqueeze(0).expand(B, -1, -1)
        else:
            t_seq = text_emb  # already (B, C, D)

        # --- Visual → Text direction ---
        # Query = visual (1 token), Key/Value = text (C tokens)
        v_normed = self.norm_v1(v_seq)
        t_normed_1 = self.norm_t1(t_seq)
        v_attn_out, _ = self.v2t_attn(v_normed, t_normed_1, t_normed_1)
        v_refined = v_seq + self.dropout(v_attn_out)  # (B, 1, D)

        # --- Text → Visual direction ---
        # Query = text (C tokens), Key/Value = visual (1 token)
        t_normed_2 = self.norm_t2(t_seq)
        v_normed_2 = self.norm_v2(v_seq)
        t_attn_out, _ = self.t2v_attn(t_normed_2, v_normed_2, v_normed_2)
        t_refined = t_seq + self.dropout(t_attn_out)  # (B, C, D)
        # Pool text side: mean over class tokens → (B, D)
        t_pooled = t_refined.mean(dim=1, keepdim=True)  # (B, 1, D)

        # Concatenate and project back to D
        fused = torch.cat([v_refined, t_pooled], dim=-1)  # (B, 1, 2D)
        fused = self.out_proj(fused.squeeze(1))  # (B, D)

        return fused
