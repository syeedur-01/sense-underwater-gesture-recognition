"""
tests/test_configs.py
=====================
Pre-flight check C2: Verify that all ablation configurations differ from
configs/sense_full.yaml strictly in their intended keys, and test chained
_base_ inheritance in load_merged_config.
"""

from pathlib import Path

import pytest
import yaml

from src.config import load_merged_config

ROOT = Path(__file__).resolve().parent.parent


def _flatten_dict(d: dict, prefix: str = "") -> dict:
    """Flatten nested dict into dot-separated keys."""
    items: dict = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, key))
        else:
            items[key] = v
    return items


def _compute_config_diff(base_cfg: dict, target_cfg: dict) -> dict[str, tuple]:
    """Return dictionary of differing dot-keys mapping to (base_val, target_val)."""
    flat_base = _flatten_dict(base_cfg)
    flat_target = _flatten_dict(target_cfg)
    diffs = {}
    all_keys = set(flat_base) | set(flat_target)
    for k in sorted(all_keys):
        v1 = flat_base.get(k)
        v2 = flat_target.get(k)
        if v1 != v2:
            diffs[k] = (v1, v2)
    return diffs


class TestConfigInheritance:
    """Test chained _base_ inheritance in load_merged_config."""

    def test_chained_base_inheritance(self, tmp_path):
        """Verify 3-level chained inheritance: grandchild -> child -> parent."""
        parent_file = tmp_path / "parent.yaml"
        child_file = tmp_path / "child.yaml"
        grandchild_file = tmp_path / "grandchild.yaml"

        parent_file.write_text(
            yaml.dump(
                {
                    "a": 1,
                    "nested": {"x": 10, "y": 20},
                }
            )
        )
        child_file.write_text(
            yaml.dump(
                {
                    "_base_": "parent.yaml",
                    "nested": {"y": 99, "z": 30},
                    "b": 2,
                }
            )
        )
        grandchild_file.write_text(
            yaml.dump(
                {
                    "_base_": "child.yaml",
                    "c": 3,
                    "nested": {"z": 999},
                }
            )
        )

        merged = load_merged_config(grandchild_file)
        assert merged["a"] == 1
        assert merged["b"] == 2
        assert merged["c"] == 3
        assert merged["nested"]["x"] == 10
        assert merged["nested"]["y"] == 99
        assert merged["nested"]["z"] == 999


class TestAblationConfigs:
    """Test that each ablation config matches sense_full.yaml except in intended keys."""

    @pytest.fixture
    def sense_full_cfg(self):
        return load_merged_config(ROOT / "configs" / "sense_full.yaml")

    def test_sense_no_enhancement_diff(self, sense_full_cfg):
        cfg = load_merged_config(ROOT / "configs" / "sense_no_enhancement.yaml")
        diff = _compute_config_diff(sense_full_cfg, cfg)
        assert set(diff.keys()) == {"data.enhancement_method"}
        assert diff["data.enhancement_method"] == ("funie_gan", "none")

    def test_sense_waternet_diff(self, sense_full_cfg):
        cfg = load_merged_config(ROOT / "configs" / "sense_waternet.yaml")
        diff = _compute_config_diff(sense_full_cfg, cfg)
        assert set(diff.keys()) == {"data.enhancement_method"}
        assert diff["data.enhancement_method"] == ("funie_gan", "waternet")

    def test_sense_mono_diff(self, sense_full_cfg):
        cfg = load_merged_config(ROOT / "configs" / "sense_mono.yaml")
        diff = _compute_config_diff(sense_full_cfg, cfg)
        assert set(diff.keys()) == {"model.stereo.enabled"}
        assert diff["model.stereo.enabled"] == (True, False)

    def test_sense_no_text_diff(self, sense_full_cfg):
        cfg = load_merged_config(ROOT / "configs" / "sense_no_text.yaml")
        diff = _compute_config_diff(sense_full_cfg, cfg)
        assert set(diff.keys()) == {"model.text_guidance.enabled"}
        assert diff["model.text_guidance.enabled"] == (True, False)

    def test_sense_neither_diff(self, sense_full_cfg):
        cfg = load_merged_config(ROOT / "configs" / "sense_neither.yaml")
        diff = _compute_config_diff(sense_full_cfg, cfg)
        assert set(diff.keys()) == {
            "model.stereo.enabled",
            "model.text_guidance.enabled",
            "data.enhancement_method",
        }
        assert diff["model.stereo.enabled"] == (True, False)
        assert diff["model.text_guidance.enabled"] == (True, False)
        assert diff["data.enhancement_method"] == ("funie_gan", "none")

    def test_sense_critical_settings_inherited(self, sense_full_cfg):
        """All ablation configs must retain critical SENSE settings from sense_full.yaml."""
        for cfg_name in [
            "sense_no_enhancement.yaml",
            "sense_waternet.yaml",
            "sense_mono.yaml",
            "sense_no_text.yaml",
            "sense_neither.yaml",
        ]:
            cfg = load_merged_config(ROOT / "configs" / cfg_name)
            assert cfg["model"]["shared_weights"] is True, f"{cfg_name} lacks shared_weights: True"
            assert (
                cfg["training"]["loss"]["class_weighted"] is True
            ), f"{cfg_name} lacks class_weighted: True"
            assert cfg["model"]["classifier"]["temperature"] == 0.07
            assert cfg["model"]["classifier"]["learnable_temperature"] is True
            assert cfg["data"]["image_size"] == 224
            assert cfg["training"]["batch_size"] == 16
            # P7.3 winner promoted into sense_full.yaml (2026-11-21): every ablation
            # must train with the winning recipe, not the old 1e-4 / 0.01 one.
            assert cfg["training"]["base_lr"] == pytest.approx(
                3e-4
            ), f"{cfg_name} lacks the promoted base_lr (3e-4)"
            assert cfg["training"]["phases"]["phase3"]["lr_scale"] == pytest.approx(
                0.3
            ), f"{cfg_name} lacks the promoted phase-3 lr_scale (0.3)"
