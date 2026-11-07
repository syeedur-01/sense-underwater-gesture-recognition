"""
src/models/text_encoder.py

P5.3 — GestureTextEncoder: pluggable text-embedding backbone for class prompts.

Design decisions:
- Backbone selected via `encoder_type` / `encoder_name` from YAML config —
  never hardcoded here.  Supported: 'sentence_bert', 'bert', 'word2vec'.
- Project default: sentence_bert / all-MiniLM-L6-v2 (same model smoke-tested
  in P0.13, already cached).
- Class names are always read from the caller (which reads DATASET_SCHEMA.md),
  never hardcoded in this file.
- Embeddings are computed once during __init__ and cached as a non-parameter
  buffer so they move with .to(device) but are excluded from optimizer updates.
- Embedding matrix shape: (num_classes, embed_dim).
"""

from __future__ import annotations

from collections import Counter

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Prompt-construction helper (Fix 3 / P5.3 spec) — unchanged from Phase 4 stub
# ---------------------------------------------------------------------------


def build_class_prompts(
    class_names: list[str],
    overrides: dict[str, str],
) -> list[str]:
    """Build a list of text prompts for each gesture class.

    Args:
        class_names: Class names ordered by class ID (0..16).
                     Source of truth: data/processed/DATASET_SCHEMA.md.
        overrides:   Map of class_name -> custom prompt string.
                     Default from configs/base.yaml:
                     ``{'true_neg': 'no gesture'}``.

    Returns:
        List of prompt strings, one per class, in class-ID order.

    Raises:
        ValueError: If an override key is not a known class name,
                    if any two prompts are identical, or if
                    ``len(result) != len(class_names)``.
    """
    unknown = set(overrides) - set(class_names)
    if unknown:
        raise ValueError(
            f"Unknown class names in overrides (not present in class_names): {unknown}"
        )

    prompts = [
        overrides[name] if name in overrides else name.replace("_", " ") for name in class_names
    ]

    # Defensive length check (should always pass for a list comprehension,
    # but required by the Fix 3 spec).
    if len(prompts) != len(class_names):
        raise ValueError(f"Expected {len(class_names)} prompts, got {len(prompts)}")

    duplicates = [p for p, count in Counter(prompts).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate prompts detected: {duplicates}")

    return prompts


# ---------------------------------------------------------------------------
# P5.3 — GestureTextEncoder
# ---------------------------------------------------------------------------

# Canonical class names ordered by gesture_id (0..16).
# Source of truth: data/processed/DATASET_SCHEMA.md.
# ID 16 (true_neg, originally label_id=-1) is remapped to 16 in manifest.csv.
CADDY_CLASS_NAMES: list[str] = [
    "start_comm",  # 0
    "end_comm",  # 1
    "up",  # 2
    "down",  # 3
    "photo",  # 4
    "backwards",  # 5
    "carry",  # 6
    "boat",  # 7
    "here",  # 8
    "mosaic",  # 9
    "num_delimiter",  # 10
    "one",  # 11
    "two",  # 12
    "three",  # 13
    "four",  # 14
    "five",  # 15
    "true_neg",  # 16
]


class GestureTextEncoder(nn.Module):
    """
    Text encoder for CADDY gesture class prompts (P5.3).

    Encodes one prompt string per class into a fixed embedding matrix
    (num_classes, embed_dim).  Embeddings are computed once at construction
    and registered as a buffer — they are frozen during training (no gradient).

    Supported encoder_type values (selected via YAML config):
      - ``sentence_bert``: SentenceTransformer model (e.g. all-MiniLM-L6-v2).
      - ``bert``:          HuggingFace BERT model (mean-pooling over token embeddings).
      - ``word2vec``:      Gensim word2vec; embedding = mean of word vectors.

    The ``encoder_name`` string is the specific pretrained model/file to load
    for the selected type.  Both are always read from config — never hardcoded.

    Args:
        class_names:   Class names ordered by gesture_id. Used to build prompts.
        overrides:     Prompt overrides (e.g. {'true_neg': 'no gesture'}).
        encoder_type:  Which text backbone to use.
        encoder_name:  Specific model name/path for the selected backbone.
        out_dim:       If not None, project embeddings to this dimension via
                       a linear layer (useful when embed_dim ≠ backbone dim).
                       If None, embeddings are used as-is.
    """

    def __init__(
        self,
        class_names: list[str] = CADDY_CLASS_NAMES,
        overrides: dict[str, str] | None = None,
        encoder_type: str = "sentence_bert",
        encoder_name: str = "all-MiniLM-L6-v2",
        out_dim: int | None = None,
    ) -> None:
        super().__init__()

        if overrides is None:
            overrides = {"true_neg": "no gesture"}

        self.class_names = class_names
        self.num_classes = len(class_names)

        # Build prompts — raises on bad overrides / duplicates
        prompts = build_class_prompts(class_names, overrides)

        # Compute raw embeddings (CPU numpy, then convert to tensor)
        raw_embeddings = self._encode(prompts, encoder_type, encoder_name)  # (C, D_raw)

        if out_dim is not None and out_dim != raw_embeddings.shape[1]:
            self.proj: nn.Linear | None = nn.Linear(raw_embeddings.shape[1], out_dim, bias=False)
            embed_dim = out_dim
        else:
            self.proj = None
            embed_dim = raw_embeddings.shape[1]

        self.embed_dim = embed_dim

        # Register as a buffer so it moves with .to(device) but is NOT a parameter
        self.register_buffer(
            "embeddings",
            torch.tensor(raw_embeddings, dtype=torch.float32),  # (C, D_raw)
        )

    @staticmethod
    def _encode(
        prompts: list[str],
        encoder_type: str,
        encoder_name: str,
    ) -> "np.ndarray":  # noqa: F821
        """Encode prompts to a numpy float32 array (num_classes, embed_dim)."""
        import numpy as np

        if encoder_type == "sentence_bert":
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(encoder_name)
            vecs = model.encode(prompts, convert_to_numpy=True, show_progress_bar=False)
            return vecs.astype(np.float32)

        elif encoder_type == "bert":
            from transformers import AutoModel, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(encoder_name)
            model = AutoModel.from_pretrained(encoder_name)
            model.eval()
            import torch

            vecs = []
            with torch.no_grad():
                for prompt in prompts:
                    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
                    outputs = model(**inputs)
                    # Mean-pool over token dimension
                    vec = outputs.last_hidden_state.mean(dim=1).squeeze(0).numpy()
                    vecs.append(vec)
            return np.stack(vecs, axis=0).astype(np.float32)

        elif encoder_type == "word2vec":
            import gensim.downloader as api

            wv = api.load(encoder_name)
            vecs = []
            for prompt in prompts:
                words = [w for w in prompt.split() if w in wv]
                if words:
                    vec = np.mean([wv[w] for w in words], axis=0)
                else:
                    vec = np.zeros(wv.vector_size, dtype=np.float32)
                vecs.append(vec)
            return np.stack(vecs, axis=0).astype(np.float32)

        else:
            raise ValueError(
                f"Unknown encoder_type: '{encoder_type}'. "
                f"Supported: 'sentence_bert', 'bert', 'word2vec'."
            )

    def forward(self) -> torch.Tensor:
        """
        Returns the class embedding matrix.

        Returns:
            Tensor of shape (num_classes, embed_dim).
            If ``out_dim`` was set, the linear projection is applied.
        """
        emb = self.embeddings  # (C, D_raw) — buffer, no grad
        if self.proj is not None:
            emb = self.proj(emb)  # (C, out_dim)
        return emb
