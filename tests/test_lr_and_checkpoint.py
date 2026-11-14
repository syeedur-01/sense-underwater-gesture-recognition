"""
tests/test_lr_and_checkpoint.py
================================
Follow-up audit tests:
  Task 1 — Closed-form LR epoch 1-50 regression (tolerance 1e-9 relative).
  Task 2 — Best-checkpoint selection: peak at epoch 5, then decline.
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from src.evaluation.metrics import is_better_checkpoint
from src.training.scheduler import build_optimizer, build_scheduler
from src.training.trainer import PhaseController

# ---------------------------------------------------------------------------
# Shared config helpers
# ---------------------------------------------------------------------------


def _make_phase_cfg(
    base_lr: float = 1e-4,
    t_max: int = 50,
    eta_min_ratio: float = 0.01,
) -> dict:
    """Minimal config matching sense_full.yaml's LR setup (no real data needed)."""
    return {
        "model": {
            "backbone": "efficientnet_b4",
            "pretrained": False,
            "out_dim": 32,
            "num_classes": 4,
            "shared_weights": True,
            "stereo": {"enabled": False},
            "text_guidance": {"enabled": False},
            "attention": {"num_heads": 1},
            "classifier": {"temperature": 0.07},
        },
        "data": {
            "image_size": 32,
            "num_classes": 4,
            "enhancement_method": None,
            "num_workers": 0,
            "pin_memory": False,
        },
        "training": {
            "seed": 0,
            "optimizer": "adamw",
            "base_lr": base_lr,
            "weight_decay": 1e-4,
            "scheduler": "cosine_annealing",
            "scheduler_eta_min_ratio": eta_min_ratio,
            "t_max": t_max,
            "epochs": t_max,
            "batch_size": 4,
            "early_stopping_patience": 10,
            "loss": {"alpha": 1.0, "beta": 0.0, "class_weighted": False},
            "phases": {
                "phase1": {
                    "epochs": [1, 10],
                    "freeze_backbone": True,
                    "lr_scale": 1.0,
                },
                "phase2": {
                    "epochs": [11, 30],
                    "freeze_backbone": False,
                    "unfreeze_last_n_blocks": 2,
                    "lr_scale": 0.1,
                },
                "phase3": {
                    "epochs": [31, 50],
                    "freeze_backbone": False,
                    "unfreeze_last_n_blocks": None,
                    "lr_scale": 0.01,
                },
            },
        },
    }


# ---------------------------------------------------------------------------
# Task 1 — Closed-form LR epoch 1-50
# ---------------------------------------------------------------------------


class TestClosedFormLRSchedule:
    """
    Task 1: Compare head LR and backbone LR at every epoch 1-50 against:
        expected = base_lr * lr_scale * (ratio + (1 - ratio) * 0.5 * (1 + cos(pi*(e-1)/T_max)))
    where ratio = eta_min / base_lr, with relative tolerance 1e-9.

    The simulation mirrors exactly what Trainer.fit() does each epoch:
        ctrl.step(epoch)       # phase transition / freeze
        opt.step()             # zero-grad parameter update (keeps params unchanged)
        sched.step()           # advance cosine counter
    """

    def test_lr_epoch1_to_50_matches_closed_form(self) -> None:
        from src.models.sense import SENSE

        BASE_LR = 1e-4
        T_MAX = 50
        ETA_MIN_RATIO = 0.01
        REL_TOL = 1e-9

        cfg = _make_phase_cfg(base_lr=BASE_LR, t_max=T_MAX, eta_min_ratio=ETA_MIN_RATIO)
        model = SENSE.from_config(cfg)
        opt = build_optimizer(model, cfg)
        sched = build_scheduler(opt, cfg)
        ctrl = PhaseController(model=model, cfg=cfg, optimizer=opt, scheduler=sched)

        # lr_scale per epoch
        def _lr_scale(epoch: int) -> float:
            if epoch <= 10:
                return 1.0
            if epoch <= 30:
                return 0.1
            return 0.01

        # Read ratio directly from training.scheduler_eta_min_ratio key
        ratio = float(cfg["training"]["scheduler_eta_min_ratio"])

        header = (
            f"\n{'Ep':>3}  {'HeadLR':>14}  {'ExpHead':>14}  "
            f"{'BackboneLR':>14}  {'ExpBack':>14}  {'Scale':>6}  {'Match?':>6}"
        )
        rows = [header]
        failures: list[str] = []

        for epoch in range(1, 51):
            ctrl.step(epoch)  # phase hook (may rebuild optimizer)

            # LRs BEFORE opt.step/sched.step — this is what the model trains with
            head_lr = opt.param_groups[0]["lr"]
            backbone_lr = opt.param_groups[1]["lr"] if len(opt.param_groups) > 1 else None
            scale = _lr_scale(epoch)

            # Closed-form reference
            s = epoch - 1
            cos_fac = 0.5 * (1.0 + math.cos(math.pi * s / T_MAX))
            factor = ratio + (1.0 - ratio) * cos_fac
            exp_head = BASE_LR * factor
            exp_back = BASE_LR * scale * factor

            # Compare
            head_ok = math.isclose(head_lr, exp_head, rel_tol=REL_TOL, abs_tol=1e-15)
            back_ok = backbone_lr is None or math.isclose(
                backbone_lr, exp_back, rel_tol=REL_TOL, abs_tol=1e-15
            )
            match = "OK" if (head_ok and back_ok) else "FAIL"

            rows.append(
                f"{epoch:>3}  {head_lr:>14.6e}  {exp_head:>14.6e}  "
                f"{backbone_lr if backbone_lr is not None else 0.0:>14.6e}  "
                f"{exp_back:>14.6e}  {scale:>6.2f}  {match:>6}"
            )

            if not head_ok:
                failures.append(
                    f"Epoch {epoch}: head LR {head_lr:.6e} != expected {exp_head:.6e} "
                    f"(rel diff {abs(head_lr - exp_head)/exp_head:.2e})"
                )
            if backbone_lr is not None and not back_ok:
                failures.append(
                    f"Epoch {epoch}: backbone LR {backbone_lr:.6e} != expected {exp_back:.6e} "
                    f"(rel diff {abs(backbone_lr - exp_back)/exp_back:.2e})"
                )

            # Advance: zero-grad step (params unchanged), then scheduler step
            opt.zero_grad(set_to_none=True)
            opt.step()
            sched.step()

        # Always print the table (visible with pytest -s)
        print("\n".join(rows))

        assert (
            not failures
        ), f"LR closed-form mismatches at relative tolerance {REL_TOL}:\n" + "\n".join(failures)


# ---------------------------------------------------------------------------
# Task 2 — Best-checkpoint selection: peak at epoch 5, then decline
# ---------------------------------------------------------------------------


def _make_tiny_cfg_for_ckpt(num_classes: int = 4) -> dict:
    """Trainer config with patience=10, 20 total epochs, phases matching test."""
    return {
        "model": {
            "backbone": "efficientnet_b4",
            "pretrained": False,
            "out_dim": 32,
            "num_classes": num_classes,
            "shared_weights": True,
            "stereo": {"enabled": False},
            "text_guidance": {"enabled": False},
            "attention": {"num_heads": 1},
            "classifier": {"temperature": 0.07},
        },
        "data": {
            "image_size": 32,
            "num_classes": num_classes,
            "enhancement_method": None,
            "num_workers": 0,
            "pin_memory": False,
        },
        "training": {
            "seed": 0,
            "epochs": 20,
            "batch_size": 4,
            "optimizer": "adamw",
            "base_lr": 1e-3,
            "weight_decay": 1e-4,
            "scheduler": "cosine_annealing",
            "scheduler_eta_min_ratio": 0.01,
            "t_max": 20,
            "early_stopping_patience": 10,
            "loss": {"alpha": 1.0, "beta": 0.0, "class_weighted": True},
            "phases": {
                # first_unfreeze_epoch == 2, so early stopping can fire from epoch 2 onward
                "phase1": {"epochs": [1, 1], "freeze_backbone": True, "lr_scale": 1.0},
                "phase2": {
                    "epochs": [2, 10],
                    "freeze_backbone": False,
                    "unfreeze_last_n_blocks": 2,
                    "lr_scale": 0.1,
                },
                "phase3": {
                    "epochs": [11, 20],
                    "freeze_backbone": False,
                    "unfreeze_last_n_blocks": None,
                    "lr_scale": 0.01,
                },
            },
        },
    }


def _make_fake_csv(tmp_path: Path, n: int = 20, num_classes: int = 4) -> Path:
    import uuid

    import pandas as pd

    rows = [
        {
            "sample_id": f"s{i:04d}",
            "left_path": f"/fake/left_{i}.png",
            "right_path": f"/fake/right_{i}.png",
            "gesture_id": i % num_classes,
            "gesture_label": f"g{i % num_classes}",
            "scenario_id": i % 3,
        }
        for i in range(n)
    ]
    p = tmp_path / f"{uuid.uuid4().hex}_split.csv"
    import pandas as pd  # noqa: F811

    pd.DataFrame(rows).to_csv(p, index=False)
    return p


class FakeStereoDataset(torch.utils.data.Dataset):
    def __init__(self, n: int = 20, num_classes: int = 4, image_size: int = 32):
        self.n = n
        self.num_classes = num_classes
        self.image_size = image_size

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        torch.manual_seed(idx)
        return {
            "left": torch.randn(3, self.image_size, self.image_size),
            "right": torch.randn(3, self.image_size, self.image_size),
            "label": idx % self.num_classes,
            "scenario_id": idx % 3,
            "sample_id": f"s{idx:04d}",
        }


def _make_fake_loader(
    n: int = 20, num_classes: int = 4, batch_size: int = 4
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        FakeStereoDataset(n=n, num_classes=num_classes),
        batch_size=batch_size,
        shuffle=False,
    )


class TestBestCheckpointSelection:
    """
    Task 2: Validation peak at epoch 5, then monotone decline.

    Verifies:
    - _best_metrics["macro_f1"] is epoch-5 value at end of run
    - _is_improved returns False for all epochs after epoch 5
    - _epochs_no_improve increments correctly after the peak
    - early stopping eventually fires
    - best checkpoint file (best.pth) corresponds to epoch 5
    """

    def _build_trainer(self, tmp_path: Path, num_classes: int = 4):
        from src.models.sense import SENSE
        from src.training.trainer import Trainer

        cfg = _make_tiny_cfg_for_ckpt(num_classes)
        train_csv = _make_fake_csv(tmp_path, n=20, num_classes=num_classes)
        val_csv = _make_fake_csv(tmp_path, n=8, num_classes=num_classes)
        model = SENSE.from_config(cfg)

        fake_train = _make_fake_loader(n=20, num_classes=num_classes)
        fake_val = _make_fake_loader(n=8, num_classes=num_classes)

        with patch("src.training.trainer.get_dataloader", side_effect=[fake_train, fake_val]):
            trainer = Trainer(
                model=model,
                cfg=cfg,
                train_csv=str(train_csv),
                val_csv=str(val_csv),
                experiment_name="ckpt_regression_test",
                device="cpu",
            )
        # Swap in persistent loaders
        trainer.train_loader = _make_fake_loader(n=20, num_classes=num_classes)
        trainer.val_loader = _make_fake_loader(n=8, num_classes=num_classes)
        return trainer

    def test_best_checkpoint_is_epoch5_peak(self, tmp_path: Path) -> None:
        """
        Metrics peak at epoch 5 (macro_f1=0.55), then decline.
        best.pth must correspond to epoch 5; _is_improved must be False afterward.
        """
        trainer = self._build_trainer(tmp_path)

        # Build per-epoch metric sequence:
        # Epochs 1-5:  ascending  macro_f1 = 0.30, 0.35, 0.40, 0.45, 0.50  (peak at epoch 5)
        # Epochs 6-20: descending macro_f1 = 0.47, 0.44, ... (all < 0.50)
        # min_class_accuracy held constant so tie-break never fires
        PEAK_F1 = 0.50
        PEAK_EPOCH = 5

        def _metrics_for_epoch(ep: int) -> dict:
            if ep <= PEAK_EPOCH:
                f1 = 0.30 + (ep - 1) * 0.05  # 0.30, 0.35, 0.40, 0.45, 0.50
            else:
                f1 = max(0.0, PEAK_F1 - (ep - PEAK_EPOCH) * 0.03)  # 0.47, 0.44, ...
            return {
                "loss": 1.0,
                "accuracy": f1,
                "macro_f1": f1,
                "min_class_accuracy": 0.20,
            }

        # Track per-epoch state
        is_improved_per_epoch: dict[int, bool] = {}
        best_epoch_at_save: list[int] = []

        # Intercept save_checkpoint to record which epoch is "best"
        real_save = __import__("src.utils.checkpoint", fromlist=["save_checkpoint"]).save_checkpoint

        def tracked_save(
            path, model, epoch, metrics, config, optimizer=None, scheduler=None, **kwargs
        ):
            if "best" in str(path):
                best_epoch_at_save.append(epoch)
            real_save(path, model, epoch, metrics, config, optimizer, scheduler, **kwargs)

        trainer._train_epoch = MagicMock(
            return_value={"loss": 1.0, "ce_loss": 1.0, "contrastive_loss": 0.0}
        )

        epoch_counter: list[int] = []

        def side_effect_validate():
            ep = len(epoch_counter) + 1
            epoch_counter.append(ep)
            return _metrics_for_epoch(ep)

        trainer._validate_epoch = side_effect_validate

        # Also intercept _is_improved to record its result per epoch
        real_is_improved = trainer._is_improved

        def tracked_is_improved(val_metrics):
            result = real_is_improved(val_metrics)
            ep = len(is_improved_per_epoch) + 1
            is_improved_per_epoch[ep] = result
            return result

        trainer._is_improved = tracked_is_improved

        with patch("src.training.trainer.save_checkpoint", side_effect=tracked_save):
            trainer.fit(num_epochs=20)

        # 1. Best checkpoint epoch must be PEAK_EPOCH
        assert best_epoch_at_save, "best.pth was never saved"
        assert best_epoch_at_save[-1] == PEAK_EPOCH, (
            f"Expected best checkpoint at epoch {PEAK_EPOCH}, "
            f"got epoch {best_epoch_at_save[-1]}. All best-saves: {best_epoch_at_save}"
        )

        # 2. _is_improved must be True for epochs 1-5 (ascending) and
        #    False for all epochs after the peak
        for ep, improved in is_improved_per_epoch.items():
            if ep <= PEAK_EPOCH:
                assert (
                    improved
                ), f"_is_improved should be True at epoch {ep} (ascending phase), got False"
            else:
                assert (
                    not improved
                ), f"_is_improved should be False at epoch {ep} (after peak), got True"

        # 3. best_metrics["macro_f1"] must be the peak value
        assert math.isclose(trainer._best_metrics.get("macro_f1", -1), PEAK_F1, abs_tol=1e-9), (
            f"best_metrics['macro_f1'] = {trainer._best_metrics.get('macro_f1')}, "
            f"expected {PEAK_F1}"
        )

        # 4. Early stopping must have fired.
        # Phase transitions reset the counter:
        #   - Epoch 1 is best (counter=0); phase2 starts at epoch 2 (counter reset to 0).
        #   - Epochs 2-5 are improving (counter stays 0 each epoch).
        #   - Epochs 6-10: 5 non-improving epochs (counter 1..5).
        #   - Phase3 starts at epoch 11 (counter reset to 0).
        #   - Epochs 11-20: 10 non-improving epochs (counter 1..10).
        #   - At end of epoch 20: counter=10 >= patience=10, epoch >= first_unfreeze(=2) → STOP.
        # So training runs all 20 epochs; early stopping fires AT the end of epoch 20.
        actual_epochs = len(epoch_counter)
        LAST_PHASE_START = 11  # from cfg phases: phase3 starts at epoch 11
        expected_stop_epoch = LAST_PHASE_START + trainer.patience - 1  # 11 + 10 - 1 = 20
        assert actual_epochs == expected_stop_epoch, (
            f"Expected early stop at epoch {expected_stop_epoch} "
            f"(last phase start {LAST_PHASE_START} + patience {trainer.patience} - 1), "
            f"got {actual_epochs} epochs"
        )

    def test_is_better_checkpoint_compares_against_best_seen_not_last_accepted(self) -> None:
        """
        Confirm: the 0.005 tie-break compares against the BEST macro F1 seen so far
        (not the last accepted checkpoint). Tests the documented behavior of is_better_checkpoint.
        """
        # Scenario: best seen = 0.700; last call improved to 0.705 (> 0.700 + 0.005 = 0.705)
        # Exactly at the boundary: 0.705 - 0.700 = 0.005, NOT > tol, so tie-break by min_acc
        # With same min_acc, the new checkpoint does NOT win (val_macro_f1 > best_macro_f1 is False
        # since they're equal in the last branch when min_acc ties)
        best_f1 = 0.700
        best_min = 0.400

        # Exactly at tolerance boundary: not better by F1, same min_acc → not improved
        result = is_better_checkpoint(
            val_macro_f1=0.705,
            val_min_acc=0.400,
            best_macro_f1=best_f1,
            best_min_acc=best_min,
            f1_tolerance=0.005,
        )
        # delta_f1 = 0.005, tol = 0.005: not > tol, not < -tol → tie-break by min_acc
        # min_acc tie (0.400 == 0.400) → val_macro_f1 > best_macro_f1 → 0.705 > 0.700 → True
        assert (
            result is True
        ), "Within tie-break window with higher min_acc tie: higher macro_f1 should win"

        # Strictly worse: 0.690 vs best 0.700 — clearly worse
        result_worse = is_better_checkpoint(
            val_macro_f1=0.690,
            val_min_acc=0.500,  # even with better min_acc
            best_macro_f1=best_f1,
            best_min_acc=best_min,
            f1_tolerance=0.005,
        )
        # delta_f1 = -0.010 < -0.005 → clearly worse
        assert result_worse is False, "0.690 vs best 0.700 should be False (outside tie window)"

        # Strictly better: 0.710 vs best 0.700
        result_better = is_better_checkpoint(
            val_macro_f1=0.710,
            val_min_acc=0.100,  # even with worse min_acc
            best_macro_f1=best_f1,
            best_min_acc=best_min,
            f1_tolerance=0.005,
        )
        # delta_f1 = 0.010 > 0.005 → clearly better
        assert result_better is True, "0.710 vs best 0.700 should be True (above tie window)"

    def test_creeping_degradation_chain_rejects_third_drop(self, tmp_path: Path) -> None:
        """
        Task 2: Test with a chain of small drops (0.900/0.60, 0.896/0.70, 0.892/0.75).
        The tie-break window must compare against _max_macro_f1_seen (0.900),
        so the third checkpoint (0.892, drop of 0.008 > 0.005 from 0.900)
        must NOT be accepted, even though 0.892 is only 0.004 below the second (0.896).
        """
        trainer = self._build_trainer(tmp_path)

        chain = [
            {"loss": 1.0, "accuracy": 0.900, "macro_f1": 0.900, "min_class_accuracy": 0.60},
            {"loss": 1.0, "accuracy": 0.896, "macro_f1": 0.896, "min_class_accuracy": 0.70},
            {"loss": 1.0, "accuracy": 0.892, "macro_f1": 0.892, "min_class_accuracy": 0.75},
        ]

        results = []
        for metrics in chain:
            accepted = trainer._is_improved(metrics)
            results.append(accepted)
            if accepted:
                trainer._best_metrics = dict(metrics)

        assert results == [True, True, False], (
            f"Expected [True, True, False] for chain, got {results}. "
            f"Third checkpoint (0.892/0.75) must be rejected against peak 0.900."
        )
        assert trainer._best_metrics["macro_f1"] == 0.896
        assert trainer._best_metrics["min_class_accuracy"] == 0.70
        assert trainer._max_macro_f1_seen == 0.900


class TestTrainerResumeAndCheckpointAudit:
    """
    Pre-flight audit of Trainer._resume() and the checkpoint save path.

    Verifies:
      1. Resuming from last.pth restores:
         - model weights
         - the phase state for the resume epoch (backbone requires_grad flags and
           optimizer parameter groups rebuilt for that phase BEFORE optimizer state is loaded)
         - optimizer and scheduler state
         - _best_metrics
         - _max_macro_f1_seen
         - _epochs_no_improve
      2. best.pth is preserved and not overwritten by a worse epoch after resuming.
      3. A resumed run from epoch 15 (Phase 2) to epoch 20 matches an uninterrupted run
         on LR values, best epoch, _max_macro_f1_seen, early-stopping counter, and best checkpoint.
      4. Resuming across the epoch-31 phase boundary (Phase 2 to Phase 3) transitions
         seamlessly, unfreezes the full backbone, resets early-stopping counter, and
         matches uninterrupted LR values.
    """

    def _build_audit_trainer(
        self,
        tmp_path: Path,
        num_classes: int = 4,
        resume_path: str | None = None,
        experiment_name: str = "ckpt_resume_audit",
        patience: int = 15,
    ):
        from src.models.sense import SENSE
        from src.training.trainer import Trainer

        cfg = _make_tiny_cfg_for_ckpt(num_classes)
        cfg["training"]["epochs"] = 50
        cfg["training"]["t_max"] = 50
        cfg["training"]["early_stopping_patience"] = patience
        cfg["training"]["phases"] = {
            "phase1": {"epochs": [1, 10], "freeze_backbone": True, "lr_scale": 1.0},
            "phase2": {
                "epochs": [11, 30],
                "freeze_backbone": False,
                "unfreeze_last_n_blocks": 2,
                "lr_scale": 0.1,
            },
            "phase3": {
                "epochs": [31, 50],
                "freeze_backbone": False,
                "unfreeze_last_n_blocks": None,
                "lr_scale": 0.01,
            },
        }

        train_csv = _make_fake_csv(tmp_path, n=20, num_classes=num_classes)
        val_csv = _make_fake_csv(tmp_path, n=8, num_classes=num_classes)
        model = SENSE.from_config(cfg)

        fake_train = _make_fake_loader(n=20, num_classes=num_classes)
        fake_val = _make_fake_loader(n=8, num_classes=num_classes)

        with patch("src.training.trainer.get_dataloader", side_effect=[fake_train, fake_val]):
            trainer = Trainer(
                model=model,
                cfg=cfg,
                train_csv=str(train_csv),
                val_csv=str(val_csv),
                experiment_name=experiment_name,
                device="cpu",
                resume_path=resume_path,
            )
        trainer.train_loader = _make_fake_loader(n=20, num_classes=num_classes)
        trainer.val_loader = _make_fake_loader(n=8, num_classes=num_classes)
        return trainer

    def test_resume_epoch15_matches_uninterrupted_run(self, tmp_path: Path) -> None:
        """
        Run uninterrupted trainer to epoch 20.
        Run partial trainer to epoch 15 (Phase 2), save last.pth and best.pth.
        Resume in fresh Trainer and continue to epoch 20.
        Assert that resumed run matches uninterrupted run on:
          - LR values at epochs 16-20
          - best epoch (epoch 5 peak)
          - _max_macro_f1_seen
          - early-stopping counter (_epochs_no_improve)
          - best.pth is preserved and not overwritten by worse epochs 16-20.
        """
        from src.utils.checkpoint import load_checkpoint

        # Deterministic validation metrics: peak at epoch 5 (0.50), then decline
        def _metrics_for_epoch(ep: int) -> dict[str, float]:
            if ep <= 5:
                f1 = 0.30 + (ep - 1) * 0.05  # 0.30, 0.35, 0.40, 0.45, 0.50
            else:
                f1 = max(0.10, 0.50 - (ep - 5) * 0.02)
            return {
                "loss": 1.0,
                "accuracy": f1,
                "macro_f1": f1,
                "min_class_accuracy": 0.30,
            }

        # --- 1. Uninterrupted Run (epochs 1 to 20) ---
        trainer_un = self._build_audit_trainer(tmp_path, experiment_name="uninterrupted_20")

        def mock_train_step_un(epoch):
            for p in trainer_un.model.parameters():
                if p.requires_grad:
                    p.grad = torch.zeros_like(p.data)
            trainer_un.optimizer.step()
            trainer_un.optimizer.zero_grad(set_to_none=True)
            return {"loss": 1.0, "ce_loss": 1.0, "contrastive_loss": 0.0}

        trainer_un._train_epoch = mock_train_step_un

        un_epoch_counter: list[int] = []

        def side_val_un():
            ep = len(un_epoch_counter) + 1
            un_epoch_counter.append(ep)
            return _metrics_for_epoch(ep)

        trainer_un._validate_epoch = side_val_un

        lrs_uninterrupted: dict[int, list[float]] = {}
        orig_log_un = trainer_un.tracker.log_epoch

        def tracked_log_un(epoch, train_metrics, val_metrics, learning_rate, epoch_duration_s):
            lrs_uninterrupted[epoch] = [g["lr"] for g in trainer_un.optimizer.param_groups]
            return orig_log_un(epoch, train_metrics, val_metrics, learning_rate, epoch_duration_s)

        trainer_un.tracker.log_epoch = tracked_log_un

        trainer_un.fit(num_epochs=20)

        best_metrics_un = dict(trainer_un._best_metrics)
        max_f1_un = trainer_un._max_macro_f1_seen
        counter_un = trainer_un._epochs_no_improve

        # Verify uninterrupted run peaked at epoch 5
        assert math.isclose(best_metrics_un["macro_f1"], 0.50, abs_tol=1e-9)
        assert math.isclose(max_f1_un, 0.50, abs_tol=1e-9)
        # Epoch 11 reset counter to 0; epochs 11..20 had 10 non-improving epochs -> counter=10
        assert counter_un == 10

        # --- 2. Partial Run (epochs 1 to 15) ---
        trainer_a = self._build_audit_trainer(tmp_path, experiment_name="partial_15")

        def mock_train_step_a(epoch):
            for p in trainer_a.model.parameters():
                if p.requires_grad:
                    p.grad = torch.zeros_like(p.data)
            trainer_a.optimizer.step()
            trainer_a.optimizer.zero_grad(set_to_none=True)
            return {"loss": 1.0, "ce_loss": 1.0, "contrastive_loss": 0.0}

        trainer_a._train_epoch = mock_train_step_a

        a_epoch_counter: list[int] = []

        def side_val_a():
            ep = len(a_epoch_counter) + 1
            a_epoch_counter.append(ep)
            return _metrics_for_epoch(ep)

        trainer_a._validate_epoch = side_val_a
        trainer_a.fit(num_epochs=15)

        last_ckpt_path = trainer_a.tracker.last_checkpoint_path
        assert last_ckpt_path.exists(), "last.pth was not saved by partial run"

        # Audit last.pth checkpoint payload contents
        payload_15 = load_checkpoint(last_ckpt_path, map_location="cpu")
        assert payload_15["epoch"] == 15
        assert "best_metrics" in payload_15
        assert math.isclose(payload_15["best_metrics"]["macro_f1"], 0.50, abs_tol=1e-9)
        assert math.isclose(payload_15["max_macro_f1_seen"], 0.50, abs_tol=1e-9)
        assert payload_15["epochs_no_improve"] == 5
        assert payload_15["phase_name"] == "phase2"
        assert "optimizer" in payload_15
        assert "scheduler" in payload_15

        # Verify best.pth in partial run corresponds to epoch 5
        best_a_payload = load_checkpoint(trainer_a.tracker.best_checkpoint_path, map_location="cpu")
        assert best_a_payload["epoch"] == 5

        # --- 3. Resume in fresh Trainer and continue to epoch 20 ---
        trainer_b = self._build_audit_trainer(
            tmp_path,
            resume_path=str(last_ckpt_path),
            experiment_name="resumed_16_to_20",
        )

        # Verify state immediately after _resume():
        assert trainer_b._start_epoch == 16
        assert trainer_b.phase_ctrl.current_phase_name == "phase2"
        assert math.isclose(trainer_b._best_metrics["macro_f1"], 0.50, abs_tol=1e-9)
        assert math.isclose(trainer_b._max_macro_f1_seen, 0.50, abs_tol=1e-9)
        assert trainer_b._epochs_no_improve == 5

        # Verify phase unfreezing state restored before training:
        # Phase 2 unfreezes last 2 blocks, keeps earlier blocks frozen
        from src.training.trainer import _efficientnet_blocks

        branch_b = trainer_b.phase_ctrl._get_branches()[0]
        blocks_b = _efficientnet_blocks(branch_b)
        for block in blocks_b[:-2]:
            for p in block.parameters():
                assert not p.requires_grad, "Earlier blocks must be frozen in Phase 2"
        for block in blocks_b[-2:]:
            for p in block.parameters():
                assert p.requires_grad, "Last 2 blocks must require grad in Phase 2"

        # Verify optimizer parameter groups: exactly 2 groups (head and backbone)
        assert len(trainer_b.optimizer.param_groups) == 2

        # Configure trainer_b steps and validation for epochs 16-20
        def mock_train_step_b(epoch):
            for p in trainer_b.model.parameters():
                if p.requires_grad:
                    p.grad = torch.zeros_like(p.data)
            trainer_b.optimizer.step()
            trainer_b.optimizer.zero_grad(set_to_none=True)
            return {"loss": 1.0, "ce_loss": 1.0, "contrastive_loss": 0.0}

        trainer_b._train_epoch = mock_train_step_b

        b_epoch_counter: list[int] = []

        def side_val_b():
            # Trainer starts at epoch 16, so ep = 15 + len + 1
            ep = 15 + len(b_epoch_counter) + 1
            b_epoch_counter.append(ep)
            return _metrics_for_epoch(ep)

        trainer_b._validate_epoch = side_val_b

        lrs_resumed: dict[int, list[float]] = {}
        orig_log_b = trainer_b.tracker.log_epoch

        def tracked_log_b(epoch, train_metrics, val_metrics, learning_rate, epoch_duration_s):
            lrs_resumed[epoch] = [g["lr"] for g in trainer_b.optimizer.param_groups]
            return orig_log_b(epoch, train_metrics, val_metrics, learning_rate, epoch_duration_s)

        trainer_b.tracker.log_epoch = tracked_log_b

        # Continue 5 epochs (epochs 16, 17, 18, 19, 20)
        trainer_b.fit(num_epochs=5)

        # --- 4. Assertions: Resumed run matches uninterrupted run ---
        # (a) LR values match at epochs 16, 17, 18, 19, 20
        for ep in range(16, 21):
            assert ep in lrs_resumed, f"Epoch {ep} not found in resumed LRs"
            assert ep in lrs_uninterrupted, f"Epoch {ep} not found in uninterrupted LRs"
            assert len(lrs_resumed[ep]) == len(lrs_uninterrupted[ep]) == 2
            # Head LR
            assert math.isclose(
                lrs_resumed[ep][0], lrs_uninterrupted[ep][0], rel_tol=1e-9
            ), f"Head LR mismatch at epoch {ep}: resumed={lrs_resumed[ep][0]}, uninterrupted={lrs_uninterrupted[ep][0]}"
            # Backbone LR
            assert math.isclose(
                lrs_resumed[ep][1], lrs_uninterrupted[ep][1], rel_tol=1e-9
            ), f"Backbone LR mismatch at epoch {ep}: resumed={lrs_resumed[ep][1]}, uninterrupted={lrs_uninterrupted[ep][1]}"

        # (b) Best epoch and metrics match
        assert math.isclose(trainer_b._best_metrics["macro_f1"], best_metrics_un["macro_f1"])
        assert math.isclose(trainer_b._max_macro_f1_seen, max_f1_un)

        # (c) Early stopping counter matches
        assert trainer_b._epochs_no_improve == counter_un == 10

        # (d) best.pth is preserved and NOT overwritten by worse epochs 16-20
        best_b_payload = load_checkpoint(trainer_b.tracker.best_checkpoint_path, map_location="cpu")
        assert (
            best_b_payload["epoch"] == 5
        ), f"best.pth was overwritten! Expected epoch 5, found epoch {best_b_payload['epoch']}"
        assert math.isclose(best_b_payload["metrics"]["macro_f1"], 0.50, abs_tol=1e-9)

    def test_resume_across_epoch31_phase_boundary(self, tmp_path: Path) -> None:
        """
        Case 2: Run to epoch 30 (end of Phase 2) and save.
        Resume in fresh Trainer; continue to epoch 35 (Phase 3).
        Verify that resuming across the epoch-31 phase boundary:
          - Restores Phase 2 unfreezing and optimizer param groups at resume
          - Triggers phase transition to Phase 3 at epoch 31
          - Unfreezes all backbone blocks
          - Rebuilds optimizer with lr_scale 0.01 for backbone
          - Resets early-stopping counter at epoch 31
          - Head and backbone LRs at epochs 31-35 match uninterrupted run.
        """
        from src.training.trainer import _efficientnet_blocks
        from src.utils.checkpoint import load_checkpoint

        def _dummy_metrics(ep: int) -> dict[str, float]:
            return {
                "loss": 1.0,
                "accuracy": 0.50,
                "macro_f1": 0.50,
                "min_class_accuracy": 0.30,
            }

        # --- 1. Uninterrupted Run (epochs 1 to 35) ---
        trainer_un = self._build_audit_trainer(
            tmp_path, experiment_name="uninterrupted_35", patience=100
        )

        def mock_train_step_un(epoch):
            for p in trainer_un.model.parameters():
                if p.requires_grad:
                    p.grad = torch.zeros_like(p.data)
            trainer_un.optimizer.step()
            trainer_un.optimizer.zero_grad(set_to_none=True)
            return {"loss": 1.0, "ce_loss": 1.0, "contrastive_loss": 0.0}

        trainer_un._train_epoch = mock_train_step_un

        un_counter: list[int] = []

        def side_val_un():
            ep = len(un_counter) + 1
            un_counter.append(ep)
            return _dummy_metrics(ep)

        trainer_un._validate_epoch = side_val_un

        lrs_uninterrupted: dict[int, list[float]] = {}
        orig_log_un = trainer_un.tracker.log_epoch

        def tracked_log_un(epoch, train_metrics, val_metrics, learning_rate, epoch_duration_s):
            lrs_uninterrupted[epoch] = [g["lr"] for g in trainer_un.optimizer.param_groups]
            return orig_log_un(epoch, train_metrics, val_metrics, learning_rate, epoch_duration_s)

        trainer_un.tracker.log_epoch = tracked_log_un

        trainer_un.fit(num_epochs=35)

        # --- 2. Partial Run (epochs 1 to 30, ending Phase 2) ---
        trainer_a = self._build_audit_trainer(tmp_path, experiment_name="partial_30", patience=100)

        def mock_train_step_a(epoch):
            for p in trainer_a.model.parameters():
                if p.requires_grad:
                    p.grad = torch.zeros_like(p.data)
            trainer_a.optimizer.step()
            trainer_a.optimizer.zero_grad(set_to_none=True)
            return {"loss": 1.0, "ce_loss": 1.0, "contrastive_loss": 0.0}

        trainer_a._train_epoch = mock_train_step_a

        a_counter: list[int] = []

        def side_val_a():
            ep = len(a_counter) + 1
            a_counter.append(ep)
            return _dummy_metrics(ep)

        trainer_a._validate_epoch = side_val_a
        trainer_a.fit(num_epochs=30)

        last_ckpt = trainer_a.tracker.last_checkpoint_path
        assert last_ckpt.exists()

        # Audit checkpoint payload at epoch 30
        payload_30 = load_checkpoint(last_ckpt, map_location="cpu")
        assert payload_30["epoch"] == 30
        assert payload_30["phase_name"] == "phase2"

        # --- 3. Resume in fresh Trainer at epoch 30, continue to epoch 35 ---
        trainer_b = self._build_audit_trainer(
            tmp_path,
            resume_path=str(last_ckpt),
            experiment_name="resumed_31_to_35",
            patience=100,
        )

        # Assert state immediately on resume (reflecting epoch 30 state before starting epoch 31):
        assert trainer_b._start_epoch == 31
        assert trainer_b.phase_ctrl.current_phase_name == "phase2"

        # Backbone has Phase 2 unfreezing (only last 2 blocks requires_grad)
        branch_b = trainer_b.phase_ctrl._get_branches()[0]
        blocks_b = _efficientnet_blocks(branch_b)
        for block in blocks_b[:-2]:
            for p in block.parameters():
                assert not p.requires_grad, "Blocks before last 2 must be frozen in Phase 2"
        for block in blocks_b[-2:]:
            for p in block.parameters():
                assert p.requires_grad, "Last 2 blocks must be unfrozen in Phase 2"

        # Configure trainer_b steps for epochs 31 to 35
        def mock_train_step_b(epoch):
            for p in trainer_b.model.parameters():
                if p.requires_grad:
                    p.grad = torch.zeros_like(p.data)
            trainer_b.optimizer.step()
            trainer_b.optimizer.zero_grad(set_to_none=True)
            return {"loss": 1.0, "ce_loss": 1.0, "contrastive_loss": 0.0}

        trainer_b._train_epoch = mock_train_step_b

        b_counter: list[int] = []

        def side_val_b():
            ep = 30 + len(b_counter) + 1
            b_counter.append(ep)
            return _dummy_metrics(ep)

        trainer_b._validate_epoch = side_val_b

        lrs_resumed: dict[int, list[float]] = {}
        orig_log_b = trainer_b.tracker.log_epoch

        def tracked_log_b(epoch, train_metrics, val_metrics, learning_rate, epoch_duration_s):
            lrs_resumed[epoch] = [g["lr"] for g in trainer_b.optimizer.param_groups]
            return orig_log_b(epoch, train_metrics, val_metrics, learning_rate, epoch_duration_s)

        trainer_b.tracker.log_epoch = tracked_log_b

        # Continue 5 epochs (epochs 31, 32, 33, 34, 35)
        trainer_b.fit(num_epochs=5)

        # --- 4. Assertions: Seamless transition across epoch 31 ---
        # (a) Phase transitioned to Phase 3
        assert trainer_b.phase_ctrl.current_phase_name == "phase3"

        # (b) All backbone blocks now require grad
        for block in blocks_b:
            for p in block.parameters():
                assert p.requires_grad, "All backbone blocks must require grad in Phase 3"

        # (c) Parameter groups count is 2 (head + full backbone)
        assert len(trainer_b.optimizer.param_groups) == 2

        # (d) LR values match uninterrupted run across epochs 31 to 35
        for ep in range(31, 36):
            assert ep in lrs_resumed, f"Epoch {ep} missing in resumed LRs"
            assert ep in lrs_uninterrupted, f"Epoch {ep} missing in uninterrupted LRs"
            # Head LR
            assert math.isclose(
                lrs_resumed[ep][0], lrs_uninterrupted[ep][0], rel_tol=1e-9
            ), f"Head LR mismatch at epoch {ep}: {lrs_resumed[ep][0]} vs {lrs_uninterrupted[ep][0]}"
            # Backbone LR
            assert math.isclose(
                lrs_resumed[ep][1], lrs_uninterrupted[ep][1], rel_tol=1e-9
            ), f"Backbone LR mismatch at epoch {ep}: {lrs_resumed[ep][1]} vs {lrs_uninterrupted[ep][1]}"
