"""
src/models/text_encoder.py
Text encoding utilities for SENSE (Phase 5).

Only the prompt-construction helper is implemented here (Fix 3 / P5.3).
Full GestureTextEncoder is Phase 5 / P5.3.

Class name ordering (by ID 0..16) comes from DATASET_SCHEMA.md:
  0: start_comm, 1: end_comm, 2: up, 3: down, 4: photo, 5: backwards,
  6: carry, 7: boat, 8: here, 9: mosaic, 10: num_delimiter,
  11: one, 12: two, 13: three, 14: four, 15: five, 16: true_neg
"""

from __future__ import annotations

from collections import Counter


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
