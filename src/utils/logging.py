"""
src/utils/logging.py
=====================
P6.5 — ExperimentTracker: creates a self-contained, reproducible experiment
        record under ``runs/{experiment_name}/{timestamp}/``.
P6.6 — TensorBoard logging: scalars for train/val loss components, accuracy,
        macro F1, min-class accuracy, learning rate, epoch timing.

Nine artifacts produced per run (P6.5 spec):
  1. config.yaml            — exact copy of the YAML config used
  2. metrics.csv            — one row per epoch (all tracked scalars)
  3. TensorBoard event file — written into the run directory
  4. training.log           — plain-text log of console output
  5. checkpoints/best.pth   — best checkpoint (highest val macro F1)
  6. checkpoints/last.pth   — latest checkpoint
  7. seed.txt               — exact seed used
  8. runtime.txt            — total wall-clock training time
  9. gpu_info.txt           — nvidia-smi -q output (empty if unavailable)
  (10. git_commit.txt       — git rev-parse HEAD, "not available" if absent)

The plan lists nine artifacts; git_commit.txt is the implicit tenth that the
plan describes under item 9 and counts together — both are always written.

Usage (wired into Trainer in P6.8)::

    tracker = ExperimentTracker(experiment_name="sense_full", cfg=cfg)
    tracker.start(seed=42)
    for epoch in range(1, num_epochs + 1):
        ...
        tracker.log_epoch(epoch, train_metrics, val_metrics, lr)
    tracker.finish()
    best_path = tracker.best_checkpoint_path
    last_path = tracker.last_checkpoint_path
"""

from __future__ import annotations

import csv
import logging
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column order in metrics.csv
# ---------------------------------------------------------------------------
_METRICS_CSV_FIELDNAMES = [
    "epoch",
    "train_loss",
    "train_ce_loss",
    "train_contrastive_loss",
    "val_loss",
    "val_accuracy",
    "val_macro_f1",
    "val_min_class_accuracy",
    "learning_rate",
    "epoch_duration_s",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _capture_gpu_info() -> str:
    """Return nvidia-smi -q output, or empty string if unavailable."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "-q"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _capture_git_commit() -> str:
    """Return git rev-parse HEAD output, or 'not available'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        commit = result.stdout.strip()
        return commit if commit else "not available"
    except Exception:
        return "not available"


# ---------------------------------------------------------------------------
# ExperimentTracker
# ---------------------------------------------------------------------------


class ExperimentTracker:
    """
    P6.5 / P6.6 — Experiment tracking + TensorBoard logging.

    Creates ``runs/{experiment_name}/{timestamp}/`` and writes all nine
    required artifacts automatically.  TensorBoard scalars are written via
    ``torch.utils.tensorboard.SummaryWriter``.

    Args:
        experiment_name: Logical name for this run (e.g. ``"sense_full"``).
        cfg:             Full YAML config dict for this run.
        base_dir:        Root directory for all runs.  Defaults to ``runs/``.
    """

    def __init__(
        self,
        experiment_name: str,
        cfg: dict[str, Any],
        base_dir: str | Path = "runs",
    ) -> None:
        self.experiment_name = experiment_name
        self.cfg = cfg
        self._start_time: float | None = None
        self._tb_writer: Any | None = None  # SummaryWriter, imported lazily
        self._csv_writer: csv.DictWriter | None = None
        self._csv_file: Any | None = None
        self._file_handler: logging.FileHandler | None = None
        self._prev_root_level: int = logging.WARNING  # restored in finish()

        # Create timestamped run directory
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_dir = Path(base_dir) / experiment_name / ts
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Subdirectory for checkpoints (best.pth / last.pth land here)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_checkpoint_path = self.checkpoint_dir / "best.pth"
        self.last_checkpoint_path = self.checkpoint_dir / "last.pth"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, seed: int) -> None:
        """
        Call once at the very beginning of training (before the first epoch).

        Writes ``config.yaml``, ``seed.txt``, ``gpu_info.txt``,
        ``git_commit.txt``, opens ``metrics.csv``, starts TensorBoard writer,
        and attaches a ``training.log`` file handler to the root logger.
        """
        self._start_time = time.monotonic()

        # 1. config.yaml
        config_path = self.run_dir / "config.yaml"
        with config_path.open("w", encoding="utf-8") as f:
            yaml.dump(self.cfg, f, default_flow_style=False, allow_unicode=True)

        # 2. seed.txt
        (self.run_dir / "seed.txt").write_text(str(seed), encoding="utf-8")

        # 3. gpu_info.txt
        (self.run_dir / "gpu_info.txt").write_text(_capture_gpu_info(), encoding="utf-8")

        # 4. git_commit.txt
        (self.run_dir / "git_commit.txt").write_text(_capture_git_commit(), encoding="utf-8")

        # 5. training.log — attach file handler to root logger
        #    Also ensure the root logger passes INFO+ records to the handler;
        #    we save the previous level and restore it in finish().
        log_path = self.run_dir / "training.log"
        self._file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        self._file_handler.setLevel(logging.DEBUG)
        fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
        self._file_handler.setFormatter(fmt)
        root_logger = logging.getLogger()
        self._prev_root_level = root_logger.level  # save for restore in finish()
        if root_logger.level == logging.NOTSET or root_logger.level > logging.INFO:
            root_logger.setLevel(logging.INFO)
        root_logger.addHandler(self._file_handler)
        logger.info("ExperimentTracker started: run_dir=%s, seed=%d", self.run_dir, seed)

        # 6. metrics.csv — open and write header
        metrics_path = self.run_dir / "metrics.csv"
        self._csv_file = metrics_path.open("w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=_METRICS_CSV_FIELDNAMES)
        self._csv_writer.writeheader()
        self._csv_file.flush()

        # 7. TensorBoard SummaryWriter (P6.6)
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore[import]

            self._tb_writer = SummaryWriter(log_dir=str(self.run_dir))
            logger.info("TensorBoard writer opened at %s", self.run_dir)
        except ImportError:
            logger.warning("tensorboard not installed; TensorBoard logging disabled.")
            self._tb_writer = None

    def finish(self) -> None:
        """
        Call once at the very end of training.

        Writes ``runtime.txt``, closes the CSV, closes TensorBoard writer,
        and removes the per-run file handler from the root logger.
        """
        # runtime.txt
        elapsed = time.monotonic() - (self._start_time or time.monotonic())
        (self.run_dir / "runtime.txt").write_text(f"{elapsed:.2f}s", encoding="utf-8")

        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None

        if self._tb_writer is not None:
            self._tb_writer.close()
            self._tb_writer = None

        if self._file_handler is not None:
            root_logger = logging.getLogger()
            root_logger.removeHandler(self._file_handler)
            # Restore root logger level to what it was before start()
            if hasattr(self, "_prev_root_level"):
                root_logger.setLevel(self._prev_root_level)
            self._file_handler.close()
            self._file_handler = None

        logger.info(
            "ExperimentTracker finished: run_dir=%s, elapsed=%.1fs",
            self.run_dir,
            elapsed,
        )

    # ------------------------------------------------------------------
    # Per-epoch logging
    # ------------------------------------------------------------------

    def log_epoch(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
        learning_rate: float,
        epoch_duration_s: float = 0.0,
    ) -> None:
        """
        Log one epoch's worth of metrics.

        Args:
            epoch:            1-indexed epoch number.
            train_metrics:    Keys: ``loss``, ``ce_loss``, ``contrastive_loss``.
            val_metrics:      Keys: ``loss``, ``accuracy``, ``macro_f1``,
                              ``min_class_accuracy``.
            learning_rate:    Current LR (after scheduler step).
            epoch_duration_s: Wall-clock seconds for this epoch.
        """
        # --- metrics.csv row ---
        row = {
            "epoch": epoch,
            "train_loss": train_metrics.get("loss", float("nan")),
            "train_ce_loss": train_metrics.get("ce_loss", float("nan")),
            "train_contrastive_loss": train_metrics.get("contrastive_loss", float("nan")),
            "val_loss": val_metrics.get("loss", float("nan")),
            "val_accuracy": val_metrics.get("accuracy", float("nan")),
            "val_macro_f1": val_metrics.get("macro_f1", float("nan")),
            "val_min_class_accuracy": val_metrics.get("min_class_accuracy", float("nan")),
            "learning_rate": learning_rate,
            "epoch_duration_s": epoch_duration_s,
        }
        if self._csv_writer is not None:
            self._csv_writer.writerow(row)
            self._csv_file.flush()

        # --- TensorBoard scalars (P6.6) ---
        if self._tb_writer is not None:
            tb = self._tb_writer
            # Train loss components
            tb.add_scalar("train/loss", row["train_loss"], epoch)
            tb.add_scalar("train/ce_loss", row["train_ce_loss"], epoch)
            tb.add_scalar("train/contrastive_loss", row["train_contrastive_loss"], epoch)
            # Val metrics
            tb.add_scalar("val/loss", row["val_loss"], epoch)
            tb.add_scalar("val/accuracy", row["val_accuracy"], epoch)
            tb.add_scalar("val/macro_f1", row["val_macro_f1"], epoch)
            tb.add_scalar("val/min_class_accuracy", row["val_min_class_accuracy"], epoch)
            # LR + timing
            tb.add_scalar("train/learning_rate", learning_rate, epoch)
            tb.add_scalar("train/epoch_duration_s", epoch_duration_s, epoch)

        logger.info(
            "Epoch %d | train_loss=%.4f | val_loss=%.4f | val_macro_f1=%.4f"
            " | val_acc=%.4f | lr=%.2e | %.1fs",
            epoch,
            row["train_loss"],
            row["val_loss"],
            row["val_macro_f1"],
            row["val_accuracy"],
            learning_rate,
            epoch_duration_s,
        )

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def copy_checkpoint(self, src: str | Path, dest: str | Path) -> None:
        """
        Copy ``src`` checkpoint into this run's checkpoint directory.

        ``dest`` should be a basename like ``"best.pth"`` or ``"last.pth"``,
        or a full path under ``self.checkpoint_dir``.
        """
        dest = Path(dest)
        if not dest.is_absolute():
            dest = self.checkpoint_dir / dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def run_directory(self) -> Path:
        """Return the run directory Path."""
        return self.run_dir

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "ExperimentTracker":
        return self

    def __exit__(self, *_: Any) -> None:
        self.finish()

    # ------------------------------------------------------------------
    # Convenience: verify all nine artifacts exist and are non-empty
    # ------------------------------------------------------------------

    def all_artifacts_present(self) -> tuple[bool, list[str]]:
        """
        Return (True, []) if all nine required artifacts exist and are non-empty,
        otherwise (False, [list of missing/empty artifact names]).

        Used by the test ``test_experiment_tracker_outputs``.
        """
        required: list[str] = [
            "config.yaml",
            "metrics.csv",
            "training.log",
            "seed.txt",
            "runtime.txt",
            "gpu_info.txt",
            "git_commit.txt",
            "checkpoints/best.pth",
            "checkpoints/last.pth",
        ]
        missing: list[str] = []
        for rel in required:
            p = self.run_dir / rel
            if not p.exists():
                missing.append(f"{rel} (missing)")
            elif p.stat().st_size == 0:
                # gpu_info.txt may be empty on a machine without nvidia-smi
                # but we still count it as present (the file exists)
                if rel != "gpu_info.txt":
                    missing.append(f"{rel} (empty)")
        # TensorBoard: at least one event file must exist in the run_dir
        tb_events = list(self.run_dir.glob("events.out.tfevents.*"))
        if not tb_events:
            missing.append("TensorBoard event file (missing)")
        return (len(missing) == 0, missing)
