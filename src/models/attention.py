"""
src/models/attention.py

P5.2 — SelfAttentionBlock: single-token self-attention + residual + LayerNorm.

Design rationale (required by plan — must be defensible in thesis):
  The fused stereo feature vector (B, D) is treated as a single-token sequence.
  Multi-head self-attention over a single token is equivalent to a learned,
  head-dependent linear projection of that token followed by a residual
  connection — it is a clean, parameter-efficient way to let the network
  reweight the feature dimensions in a data-driven manner before cross-modal
  fusion, without introducing a full transformer encoder layer's complexity.
  The residual connection + LayerNorm combination follows the Pre-LN convention
  (LayerNorm applied before the attention sub-layer), which is empirically more
  stable during training than Post-LN.

  Alternative considered and rejected: a simple 2-layer MLP (channel mixing
  only, no head-wise competition). The attention formulation makes the
  per-dimension selection mechanism explicit and architecturally legible,
  which matters for the thesis writeup.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SelfAttentionBlock(nn.Module):
    """
    Single-token self-attention refinement block (P5.2).

    Applies multi-head self-attention to a *single* feature vector
    (treated as a 1-token sequence), followed by a residual connection
    and LayerNorm.  The block is inserted between ``BiChannelEncoder``
    and ``CrossModalFusion``.

    Architecture (Pre-LN):
        x → LayerNorm → MultiheadAttention (Q=K=V=x) → dropout → x + residual

    Args:
        embed_dim: Feature dimension (must equal BiChannelEncoder.feature_dim).
        num_heads: Number of attention heads. Must divide ``embed_dim`` evenly.
        dropout:   Dropout probability on the attention output.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,  # (B, T, D) convention
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature tensor (B, D).

        Returns:
            Refined feature tensor (B, D) — same shape as input.
        """
        # Treat the single feature vector as a 1-token sequence: (B, 1, D)
        x_seq = x.unsqueeze(1)  # (B, 1, D)

        # Pre-LN: normalize before attention
        normed = self.norm(x_seq)

        # Self-attention (Q = K = V = normed)
        attn_out, _ = self.attn(normed, normed, normed)  # (B, 1, D)

        # Residual connection + drop
        refined = x_seq + self.dropout(attn_out)  # (B, 1, D)

        return refined.squeeze(1)  # (B, D)
