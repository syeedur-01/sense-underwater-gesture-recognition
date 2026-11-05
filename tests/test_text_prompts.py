"""
tests/test_text_prompts.py
Fix 3 verification for build_class_prompts (src/models/text_encoder.py).
"""

import pytest

from src.models.text_encoder import build_class_prompts

# Class names ordered by ID 0..16 — source: data/processed/DATASET_SCHEMA.md
CLASS_NAMES_17 = [
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

DEFAULT_OVERRIDES = {"true_neg": "no gesture"}


def test_17_names_give_17_prompts():
    """Fix 1 + Fix 3: exactly 17 prompts out for 17 class names in."""
    prompts = build_class_prompts(CLASS_NAMES_17, DEFAULT_OVERRIDES)
    assert len(prompts) == 17


def test_true_neg_override():
    """Fix 3: true_neg (ID 16) maps to 'no gesture' via override."""
    prompts = build_class_prompts(CLASS_NAMES_17, DEFAULT_OVERRIDES)
    assert prompts[16] == "no gesture"


def test_start_comm_default():
    """Fix 3: underscores replaced by spaces for non-overridden names."""
    prompts = build_class_prompts(CLASS_NAMES_17, DEFAULT_OVERRIDES)
    assert prompts[0] == "start comm"


def test_num_delimiter_default():
    """Fix 3: multi-underscore name also replaced correctly."""
    prompts = build_class_prompts(CLASS_NAMES_17, DEFAULT_OVERRIDES)
    assert prompts[10] == "num delimiter"


def test_unknown_override_raises():
    """Fix 3: unknown override key raises ValueError."""
    with pytest.raises(ValueError, match="Unknown class names"):
        build_class_prompts(CLASS_NAMES_17, {"nonexistent_class": "oops"})


def test_duplicate_prompts_raise():
    """Fix 3: two prompts resolving to the same string raise ValueError."""
    # Override 'up' so it collides with the default for 'down'
    with pytest.raises(ValueError, match="Duplicate prompts"):
        build_class_prompts(CLASS_NAMES_17, {"up": "down", "true_neg": "no gesture"})


def test_empty_overrides_still_gives_17_prompts():
    """Fix 3: no overrides → all defaults → still 17 distinct prompts."""
    prompts = build_class_prompts(CLASS_NAMES_17, {})
    assert len(prompts) == 17
    # true_neg without override becomes "true neg"
    assert prompts[16] == "true neg"
