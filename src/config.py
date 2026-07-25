"""
src/config.py
Centralized configuration loader for the SENSE project.

All paths and settings are read from environment variables (via .env)
and YAML experiment configs. Never hardcode paths anywhere else —
always import from this module.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
import yaml

# Load .env from the project root (works regardless of CWD if
# the file is placed at the project root)
load_dotenv()


class Paths:
    """Resolved filesystem paths for all project resources."""

    CADDY_ROOT: Path = Path(os.getenv("CADDY_ROOT", "./data/raw/caddy"))
    CADDY_CSV_POSITIVES: Path = Path(
        os.getenv("CADDY_CSV_POSITIVES", "./data/raw/caddy/caddy_true_positives.csv")
    )
    CADDY_CSV_NEGATIVES: Path = Path(
        os.getenv("CADDY_CSV_NEGATIVES", "./data/raw/caddy/caddy_true_negatives.csv")
    )
    ENHANCED_CACHE_DIR: Path = Path(
        os.getenv("ENHANCED_CACHE_DIR", "./data/processed/enhanced")
    )
    SPLITS_DIR: Path = Path(os.getenv("SPLITS_DIR", "./data/processed/splits"))
    FUNIE_GAN_WEIGHTS: Path = Path(
        os.getenv("FUNIE_GAN_WEIGHTS", "./weights/funie_gan/generator.pth")
    )
    WATERNET_WEIGHTS: Path = Path(
        os.getenv("WATERNET_WEIGHTS", "./weights/waternet/waternet.pth")
    )
    RUNS_DIR: Path = Path(os.getenv("RUNS_DIR", "./runs"))
    CHECKPOINTS_DIR: Path = Path(os.getenv("CHECKPOINTS_DIR", "./checkpoints"))


SEED: int = int(os.getenv("SEED", 42))
DEVICE: str = os.getenv("DEVICE", "cuda")


def load_yaml_config(path: str) -> dict:
    """Load and return a YAML config file as a Python dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)
