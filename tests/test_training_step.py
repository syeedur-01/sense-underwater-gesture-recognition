"""
tests/test_training_step.py
===========================
Tests for Phase 6 training infrastructure.
Stage A: P6.1 (SenseLoss, ContrastiveLoss) & P6.2 (Optimizer & Scheduler factory).
Stage B: P6.3 (PhaseController) & P6.4 (Checkpoint).
Stage C: P6.5 (ExperimentTracker) & P6.6 (TensorBoard) & P6.7 (Seed control).
Stage D: P6.8 (Trainer) & P6.9 (scripts/train.py).
Stage E: divergence guard — fail-fast abort on NaN/inf (P6.8 hardening).
"""

from __future__ import annotations

import copy
import math
import subprocess
import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from src.data.splits import compute_class_weights
from src.training.losses import ContrastiveLoss, SenseLoss
from src.training.scheduler import (
    build_optimizer,
    build_optimizer_and_scheduler,
    build_scheduler,
)
from src.training.trainer import PhaseController, Trainer
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.logging import ExperimentTracker
from src.utils.seed import set_seed


class TestContrastiveLoss:
    def test_contrastive_loss_scalar_and_finite(self):
        loss_fn = ContrastiveLoss(temperature=0.07)
        batch_size = 16
        num_classes = 17

        logits = torch.randn(batch_size, num_classes, requires_grad=True)
        targets = torch.randint(0, num_classes, (batch_size,))

        loss = loss_fn(logits, targets)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() > 0.0
        assert torch.isfinite(loss).item()

        # Backward pass
        loss.backward()
        assert logits.grad is not None
        assert torch.all(torch.isfinite(logits.grad))

    def test_contrastive_with_explicit_embeddings(self):
        loss_fn = ContrastiveLoss(temperature=0.07)
        batch_size = 8
        feat_dim = 64
        num_classes = 17

        visual = torch.randn(batch_size, feat_dim, requires_grad=True)
        text_emb = torch.randn(num_classes, feat_dim)
        targets = torch.randint(0, num_classes, (batch_size,))

        loss = loss_fn(visual, targets, text_embeddings=text_emb)
        assert loss.item() > 0.0
        loss.backward()
        assert visual.grad is not None

    def test_single_sample_batch_edge_case(self):
        loss_fn = ContrastiveLoss()
        logits = torch.randn(1, 17, requires_grad=True)
        targets = torch.tensor([0])
        loss = loss_fn(logits, targets)
        assert loss.item() == 0.0


class TestSenseLoss:
    def test_positive_finite_and_backward(self):
        loss_fn = SenseLoss(alpha=0.7, beta=0.3)
        batch_size = 16
        num_classes = 17

        logits = torch.randn(batch_size, num_classes, requires_grad=True)
        targets = torch.randint(0, num_classes, (batch_size,))

        loss = loss_fn(logits, targets)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0.0
        assert torch.isfinite(loss).item()

        loss.backward()
        assert logits.grad is not None
        assert torch.all(torch.isfinite(logits.grad))

    def test_weighted_ce_numerically_different_from_unweighted(self):
        """
        P6.1 Verification:
        On a synthetic imbalanced batch, weighted CE produces a numerically
        different loss value than unweighted CE, confirming the weight tensor
        is actually applied rather than silently ignored.
        """
        num_classes = 17
        batch_size = 32

        # Artificial class weights favoring minority classes
        weights = torch.ones(num_classes)
        weights[0] = 0.1  # frequent
        weights[15] = 10.0  # rare

        loss_unweighted = SenseLoss(alpha=1.0, beta=0.0, class_weights=None)
        loss_weighted = SenseLoss(alpha=1.0, beta=0.0, class_weights=weights)

        torch.manual_seed(42)
        logits = torch.randn(batch_size, num_classes)
        # Heavily biased towards class 15 to trigger distinct weighting
        targets = torch.randint(0, 2, (batch_size,))
        targets[targets == 1] = 15

        val_unweighted = loss_unweighted(logits, targets).item()
        val_weighted = loss_weighted(logits, targets).item()

        assert abs(val_weighted - val_unweighted) > 1e-3, (
            f"Weighted CE ({val_weighted:.4f}) should differ numerically "
            f"from unweighted CE ({val_unweighted:.4f})"
        )

    def test_class_weights_computed_from_training_csv(self, tmp_path: Path):
        """
        Required test:
        The weights used by the loss are computed from that run's training CSV.
        """
        csv_path = tmp_path / "mock_train.csv"
        # Create mock training CSV with known distribution
        rows = []
        # Class 0: 100 samples, Class 1: 10 samples
        for _ in range(100):
            rows.append({"sample_id": "s", "gesture_id": 0, "gesture_label": "zero"})
        for _ in range(10):
            rows.append({"sample_id": "s", "gesture_id": 1, "gesture_label": "one"})
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        cfg = {
            "training": {"loss": {"alpha": 0.7, "beta": 0.3, "class_weighted": True}},
            "model": {"text_guidance": {"enabled": True}, "num_classes": 2},
            "data": {"num_classes": 2},
        }

        loss_fn = SenseLoss.from_config(cfg, train_csv=csv_path)

        assert loss_fn.class_weights is not None
        # Expected class weights from compute_class_weights(csv_path, num_classes=2)
        expected_weights = compute_class_weights(csv_path, num_classes=2)
        assert torch.allclose(loss_fn.class_weights, expected_weights)
        # Class 1 is rare, should have higher weight than Class 0
        assert loss_fn.class_weights[1] > loss_fn.class_weights[0]

    def test_degrades_to_ce_only_when_text_disabled(self):
        """
        P6.1 specification:
        Gracefully degrades to CE-only when text_guidance.enabled=False,
        independent of the class-weighting setting.
        """
        loss_fn = SenseLoss(alpha=0.7, beta=0.3, text_enabled=False)
        assert loss_fn.beta == 0.0
        assert loss_fn.alpha == 1.0
        assert loss_fn.contrastive is None

        logits = torch.randn(8, 17)
        targets = torch.randint(0, 17, (8,))

        ce_standard = nn.CrossEntropyLoss()(logits, targets)
        loss_output = loss_fn(logits, targets)

        assert pytest.approx(loss_output.item(), abs=1e-6) == ce_standard.item()

    def test_return_components_dict(self):
        loss_fn = SenseLoss(alpha=0.7, beta=0.3)
        logits = torch.randn(8, 17)
        targets = torch.randint(0, 17, (8,))

        total_loss, metrics = loss_fn(logits, targets, return_components=True)
        assert "loss" in metrics
        assert "ce_loss" in metrics
        assert "contrastive_loss" in metrics
        assert pytest.approx(metrics["loss"], abs=1e-5) == total_loss.item()


class TestOptimizerAndScheduler:
    def test_build_optimizer_adamw(self):
        linear = nn.Linear(10, 2)
        cfg = {
            "training": {
                "optimizer": "adamw",
                "base_lr": 2.5e-4,
                "weight_decay": 5.0e-4,
            }
        }
        optimizer = build_optimizer(linear, cfg)
        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.defaults["lr"] == 2.5e-4
        assert optimizer.defaults["weight_decay"] == 5.0e-4

    def test_build_scheduler_standalone(self):
        linear = nn.Linear(10, 2)
        cfg_opt = {"training": {"optimizer": "adamw", "base_lr": 1e-4}}
        optimizer = build_optimizer(linear, cfg_opt)

        cfg_sched = {"training": {"scheduler": "step", "step_size": 5, "gamma": 0.5}}
        scheduler = build_scheduler(optimizer, cfg_sched)
        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)

    def test_scheduler_lr_trends_downward(self):
        """
        P6.2 Verification:
        LR trends downward over dummy iterations.
        """
        linear = nn.Linear(10, 2)
        cfg = {
            "training": {
                "optimizer": "adamw",
                "base_lr": 1e-3,
                "scheduler": "cosine_annealing",
                "t_max": 20,
                "min_lr": 1e-6,
            }
        }
        optimizer, scheduler = build_optimizer_and_scheduler(linear, cfg, total_epochs=20)

        lrs = []
        for _ in range(20):
            current_lr = scheduler.get_last_lr()[0]
            lrs.append(current_lr)
            # Step optimizer & scheduler
            optimizer.step()
            scheduler.step()

        # Initial LR should equal base_lr
        assert lrs[0] == 1e-3
        # Final LR should be smaller than initial LR
        assert lrs[-1] < lrs[0]
        # Check downward trend across iterations
        for i in range(len(lrs) - 1):
            assert lrs[i] >= lrs[i + 1]


# ===========================================================================
# Stage B — P6.3 (PhaseController) & P6.4 (Checkpoint) tests
# ===========================================================================

# ---------------------------------------------------------------------------
# Minimal thin SENSE-like model for PhaseController tests
# (uses actual BiChannelEncoder so the freeze logic targets the real structure)
# ---------------------------------------------------------------------------


def _make_minimal_cfg(
    freeze_backbone_p1: bool = True,
    unfreeze_p2: int = 2,
    unfreeze_p3: None = None,
) -> dict:
    """Return a minimal config dict matching base.yaml's structure."""
    return {
        "model": {
            "backbone": "efficientnet_b4",
            "pretrained": False,  # no download in tests
            "out_dim": 64,  # small for speed
            "num_classes": 17,
            "shared_weights": True,
            "stereo": {"enabled": True},
            "text_guidance": {"enabled": False},
            "attention": {"num_heads": 1},
            "classifier": {"temperature": 0.07},
        },
        "data": {"image_size": 224, "num_classes": 17},
        "training": {
            "optimizer": "adamw",
            "base_lr": 1e-4,
            "weight_decay": 1e-4,
            "epochs": 50,
            "scheduler": "cosine_annealing",
            "scheduler_eta_min_ratio": 0.01,
            "t_max": 50,
            "loss": {"alpha": 0.7, "beta": 0.0, "class_weighted": False},
            "phases": {
                "phase1": {
                    "epochs": [1, 10],
                    "freeze_backbone": freeze_backbone_p1,
                    "lr_scale": 1.0,
                },
                "phase2": {
                    "epochs": [11, 30],
                    "freeze_backbone": False,
                    "unfreeze_last_n_blocks": unfreeze_p2,
                    "lr_scale": 0.1,
                },
                "phase3": {
                    "epochs": [31, 50],
                    "freeze_backbone": False,
                    "unfreeze_last_n_blocks": unfreeze_p3,
                    "lr_scale": 0.01,
                },
            },
        },
    }


@pytest.fixture(scope="module")
def sense_model_and_cfg():
    """Build a SENSE model (stereo=True, text=False) for PhaseController tests."""
    from src.models.sense import SENSE

    cfg = _make_minimal_cfg()
    model = SENSE.from_config(cfg)
    model.eval()
    return model, cfg


class TestPhaseController:
    """P6.3 — Progressive unfreezing controller tests."""

    def test_phase1_epoch1_backbone_frozen(self, sense_model_and_cfg):
        """
        Required test: requires_grad states correct at epoch 1 (phase 1).
        Backbone params must all have requires_grad=False.
        Head/projection params must have requires_grad=True.
        """
        model, cfg = sense_model_and_cfg
        ctrl = PhaseController.from_config(model, cfg)

        transitioned = ctrl.step(epoch=1)
        assert transitioned, "Phase 1 should trigger on epoch 1"
        assert ctrl.current_phase_name == "phase1"

        # Backbone (branch[0] = EfficientNet inside left_branch) must be frozen
        backbone = model.encoder.left_branch[0]
        for name, p in backbone.named_parameters():
            assert not p.requires_grad, f"Backbone param '{name}' should be frozen in phase 1"

        # At least one non-backbone param must be trainable
        backbone_param_ids = {id(p) for p in backbone.parameters()}
        trainable_non_backbone = [
            p for p in model.parameters() if p.requires_grad and id(p) not in backbone_param_ids
        ]
        assert (
            len(trainable_non_backbone) > 0
        ), "Head/fusion/attention params should be trainable in phase 1"

    def test_phase2_epoch11_last_n_blocks_unfrozen(self, sense_model_and_cfg):
        """
        Required test: requires_grad states correct at epoch 11 (phase 2).
        Last 2 EfficientNet blocks should be unfrozen; earlier blocks stay frozen.
        """
        from src.training.trainer import _efficientnet_blocks

        model, cfg = sense_model_and_cfg
        ctrl = PhaseController.from_config(model, cfg)

        ctrl.step(epoch=1)  # → phase 1
        transitioned = ctrl.step(epoch=11)  # → phase 2
        assert transitioned, "Phase 2 should trigger on epoch 11"
        assert ctrl.current_phase_name == "phase2"

        branch = model.encoder.left_branch
        blocks = _efficientnet_blocks(branch)
        n_unfreeze = 2

        # Frozen blocks (all but last n)
        frozen_blocks = blocks[:-n_unfreeze]
        unfrozen_blocks = blocks[-n_unfreeze:]

        for blk in frozen_blocks:
            for name, p in blk.named_parameters():
                assert not p.requires_grad, (
                    f"Block param '{name}' should still be frozen in phase 2 "
                    f"(it is not in the last {n_unfreeze} blocks)"
                )

        for blk in unfrozen_blocks:
            for name, p in blk.named_parameters():
                assert p.requires_grad, (
                    f"Block param '{name}' should be unfrozen in phase 2 "
                    f"(it is in the last {n_unfreeze} blocks)"
                )

    def test_phase3_epoch31_all_backbone_unfrozen(self, sense_model_and_cfg):
        """
        Required test: requires_grad states correct at epoch 31 (phase 3).
        Entire backbone must have requires_grad=True.
        """
        model, cfg = sense_model_and_cfg
        ctrl = PhaseController.from_config(model, cfg)

        ctrl.step(epoch=1)
        ctrl.step(epoch=11)
        transitioned = ctrl.step(epoch=31)
        assert transitioned, "Phase 3 should trigger on epoch 31"
        assert ctrl.current_phase_name == "phase3"

        backbone = model.encoder.left_branch[0]
        for name, p in backbone.named_parameters():
            assert p.requires_grad, f"Backbone param '{name}' should be unfrozen in phase 3"

    def test_no_transition_within_same_phase(self, sense_model_and_cfg):
        """step() returns False when epoch stays in the same phase."""
        model, cfg = sense_model_and_cfg
        ctrl = PhaseController.from_config(model, cfg)

        ctrl.step(epoch=1)  # phase 1 — transition
        result = ctrl.step(epoch=5)  # still phase 1 — no transition
        assert not result

    def test_optimizer_groups_rebuilt_at_phase_transition(self, sense_model_and_cfg):
        """
        Required test: optimizer parameter groups change at phase transitions,
        and the newly unfrozen parameters actually receive gradient updates.
        """
        model, cfg = sense_model_and_cfg
        ctrl = PhaseController.from_config(model, cfg)

        # Phase 1: only head params in optimizer
        ctrl.step(epoch=1)
        p1_groups = [len(g["params"]) for g in ctrl.optimizer.param_groups]
        total_p1 = sum(p1_groups)

        # Phase 2: some backbone params added
        ctrl.step(epoch=11)
        p2_groups = [len(g["params"]) for g in ctrl.optimizer.param_groups]
        total_p2 = sum(p2_groups)

        # Phase 2 should have more (or equal) trainable params than Phase 1
        assert (
            total_p2 >= total_p1
        ), f"Phase 2 ({total_p2} params) should have >= Phase 1 ({total_p1} params)"

    def test_unfrozen_backbone_params_receive_gradients(self, sense_model_and_cfg):
        """
        Required test: newly unfrozen params receive gradient updates after
        a backward pass.
        """
        from src.training.trainer import _efficientnet_blocks

        model, cfg = sense_model_and_cfg
        ctrl = PhaseController.from_config(model, cfg)

        # Jump straight to phase 2
        ctrl.step(epoch=1)
        ctrl.step(epoch=11)

        model.train()

        # Run a forward/backward through the encoder directly
        x = torch.randn(2, 3, 64, 64)
        feat = model.encoder(x, x)  # stereo: returns (B, 2*out_dim)
        loss = feat.sum()
        loss.backward()

        # Last 2 blocks of left_branch backbone should have non-None, non-zero grads
        branch = model.encoder.left_branch
        blocks = _efficientnet_blocks(branch)
        unfrozen_blocks = blocks[-2:]
        for blk in unfrozen_blocks:
            for p in blk.parameters():
                if p.requires_grad:
                    assert (
                        p.grad is not None
                    ), "Unfrozen backbone param has no gradient after backward"

        model.eval()

    def test_lr_scale_applied_to_backbone_group(self, sense_model_and_cfg):
        """Backbone param group should have lr = base_lr * lr_scale."""
        model, cfg = sense_model_and_cfg
        ctrl = PhaseController.from_config(model, cfg)

        # Phase 2: lr_scale = 0.1
        ctrl.step(epoch=1)
        ctrl.step(epoch=11)

        base_lr = cfg["training"]["base_lr"]  # 1e-4
        lr_scale = 0.1

        # Backbone group has lr = base_lr * lr_scale
        backbone_lrs = [
            g["lr"]
            for g in ctrl.optimizer.param_groups
            if any(
                id(p) in {id(q) for q in model.encoder.left_branch[0].parameters()}
                for p in g["params"]
            )
        ]
        if backbone_lrs:  # only check if backbone group exists
            for lr in backbone_lrs:
                assert abs(lr - base_lr * lr_scale) < 1e-9, (
                    f"Backbone LR {lr} != base_lr * lr_scale "
                    f"({base_lr} * {lr_scale} = {base_lr * lr_scale})"
                )


class TestCheckpoint:
    """P6.4 — Checkpoint save/load round-trip tests."""

    def _make_simple_model(self) -> nn.Module:
        return nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2))

    def test_save_and_load_weights_bit_identical(self, tmp_path):
        """
        Required test: Bit-identical weights after a save/load round-trip.
        """
        model = self._make_simple_model()
        cfg = {"training": {"base_lr": 1e-4}}
        metrics = {"macro_f1": 0.85, "accuracy": 0.90, "min_class_accuracy": 0.70}

        ckpt_path = tmp_path / "test_ckpt.pth"
        save_checkpoint(ckpt_path, model, epoch=5, metrics=metrics, config=cfg)

        # Create a fresh model, load into it
        model2 = self._make_simple_model()
        payload = load_checkpoint(ckpt_path, model=model2, map_location="cpu")

        # Compare all parameter tensors
        for (n1, p1), (n2, p2) in zip(model.state_dict().items(), model2.state_dict().items()):
            assert n1 == n2
            assert torch.equal(p1, p2), f"Parameter '{n1}' differs after round-trip"

        assert payload["epoch"] == 5
        assert payload["metrics"]["macro_f1"] == pytest.approx(0.85)

    def test_checkpoint_persists_all_required_fields(self, tmp_path):
        """Checkpoint must store model, optimizer, scheduler, epoch, metrics, config."""
        model = self._make_simple_model()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        metrics = {
            "macro_f1": 0.75,
            "accuracy": 0.80,
            "min_class_accuracy": 0.60,
        }
        cfg = {"training": {"base_lr": 1e-3}, "data": {"image_size": 224}}

        ckpt_path = tmp_path / "full_ckpt.pth"
        save_checkpoint(
            ckpt_path,
            model,
            epoch=12,
            metrics=metrics,
            config=cfg,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        payload = load_checkpoint(ckpt_path, map_location="cpu")
        assert "model" in payload
        assert "optimizer" in payload
        assert "scheduler" in payload
        assert payload["epoch"] == 12
        assert payload["metrics"]["accuracy"] == pytest.approx(0.80)
        assert payload["metrics"]["min_class_accuracy"] == pytest.approx(0.60)
        assert payload["config"]["data"]["image_size"] == 224

    def test_checkpoint_round_trip_gives_identical_validation_result(self, tmp_path):
        """
        Required test: resumed state gives an identical validation result.
        After saving optimizer + scheduler, restoring them and running another
        step must produce the same gradients.
        """
        torch.manual_seed(0)
        model = self._make_simple_model()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

        # Run one step
        x = torch.randn(4, 8)
        y = torch.randint(0, 2, (4,))
        loss = nn.CrossEntropyLoss()(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Save
        ckpt_path = tmp_path / "roundtrip.pth"
        save_checkpoint(
            ckpt_path,
            model,
            epoch=1,
            metrics={"macro_f1": 0.5, "accuracy": 0.5, "min_class_accuracy": 0.0},
            config={},
            optimizer=optimizer,
            scheduler=scheduler,
        )

        # Clone state before reload
        weights_before = copy.deepcopy(model.state_dict())

        # Restore into a fresh model
        model2 = self._make_simple_model()
        optimizer2 = optim.AdamW(model2.parameters(), lr=1e-3)
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=5)
        load_checkpoint(
            ckpt_path,
            model=model2,
            optimizer=optimizer2,
            scheduler=scheduler2,
            map_location="cpu",
        )

        # Weights must match
        for (n1, p1), (n2, p2) in zip(weights_before.items(), model2.state_dict().items()):
            assert torch.equal(p1, p2), f"Param '{n1}' differs after round-trip"

        # Run another step on both — results must match
        torch.manual_seed(1)
        x2 = torch.randn(4, 8)
        y2 = torch.randint(0, 2, (4,))

        loss1 = nn.CrossEntropyLoss()(model(x2), y2)
        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()

        loss2 = nn.CrossEntropyLoss()(model2(x2), y2)
        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()

        assert torch.isclose(
            loss1, loss2, atol=1e-6
        ), f"Loss after resumed step differs: {loss1.item()} vs {loss2.item()}"

    def test_load_missing_file_raises(self, tmp_path):
        """load_checkpoint raises FileNotFoundError for a missing path."""
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            load_checkpoint(tmp_path / "nonexistent.pth")

    def test_config_deep_copy_does_not_alias(self, tmp_path):
        """Mutating config after save should not corrupt the checkpoint."""
        model = self._make_simple_model()
        cfg = {"training": {"base_lr": 1e-4}}
        ckpt_path = tmp_path / "alias_test.pth"
        save_checkpoint(
            ckpt_path,
            model,
            epoch=1,
            metrics={"macro_f1": 0.5, "accuracy": 0.5, "min_class_accuracy": 0.0},
            config=cfg,
        )

        # Mutate the original config after saving
        cfg["training"]["base_lr"] = 999.0

        payload = load_checkpoint(ckpt_path, map_location="cpu")
        assert (
            payload["config"]["training"]["base_lr"] == 1e-4
        ), "Config in checkpoint should not reflect post-save mutations"


# ===========================================================================
# Stage C — P6.5 (ExperimentTracker), P6.6 (TensorBoard), P6.7 (Seed) tests
# ===========================================================================


class TestSeedControl:
    """P6.7 — Seed reproducibility tests."""

    def test_same_seed_same_loss(self):
        """
        Required test: identical epoch-1 loss across two runs with the same seed.
        """

        def make_model():
            return nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 4))

        loss_fn = nn.CrossEntropyLoss()

        def one_epoch_loss(seed: int) -> float:
            set_seed(seed)
            model = make_model()
            opt = optim.SGD(model.parameters(), lr=0.1)
            x = torch.randn(8, 16)
            y = torch.randint(0, 4, (8,))
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            return loss.item()

        loss_a = one_epoch_loss(seed=99)
        loss_b = one_epoch_loss(seed=99)
        assert loss_a == loss_b, f"Same seed must produce same loss: {loss_a} vs {loss_b}"

    def test_different_seeds_different_loss(self):
        """Different seeds should (almost always) produce different losses."""

        def make_model():
            return nn.Sequential(nn.Linear(32, 8), nn.ReLU(), nn.Linear(8, 4))

        loss_fn = nn.CrossEntropyLoss()

        def one_loss(seed: int) -> float:
            set_seed(seed)
            model = make_model()
            x = torch.randn(8, 32)
            y = torch.randint(0, 4, (8,))
            return loss_fn(model(x), y).item()

        loss_a = one_loss(seed=1)
        loss_b = one_loss(seed=2)
        # This could theoretically collide, but essentially never will
        assert loss_a != loss_b

    def test_set_seed_sets_cudnn_flags(self):
        """set_seed must set cudnn.deterministic=True and benchmark=False."""
        import torch.backends.cudnn as cudnn

        set_seed(42)
        assert cudnn.deterministic is True
        assert cudnn.benchmark is False


class TestExperimentTracker:
    """P6.5 / P6.6 — ExperimentTracker artifact tests."""

    def _dummy_cfg(self) -> dict:
        return {
            "model": {"num_classes": 17},
            "data": {"image_size": 224},
            "training": {"seed": 42, "epochs": 1, "base_lr": 1e-4},
        }

    def test_all_nine_artifacts_present_after_1epoch_run(self, tmp_path):
        """
        P6.5 Verification:
        After a 1-epoch dummy run, all nine artifacts listed in P6.5 must
        exist and be non-empty in the run directory.
        """
        cfg = self._dummy_cfg()
        tracker = ExperimentTracker(
            experiment_name="test_sense_full",
            cfg=cfg,
            base_dir=tmp_path / "runs",
        )
        # Simulate a full 1-epoch run
        tracker.start(seed=42)

        # Fake 1-epoch checkpoint: write dummy .pth files so the
        # artifact-presence check passes for best.pth / last.pth
        dummy_state = {"epoch": 1, "model": {}, "metrics": {}, "config": cfg}
        torch.save(dummy_state, tracker.best_checkpoint_path)
        torch.save(dummy_state, tracker.last_checkpoint_path)

        tracker.log_epoch(
            epoch=1,
            train_metrics={"loss": 1.5, "ce_loss": 1.0, "contrastive_loss": 0.5},
            val_metrics={
                "loss": 1.4,
                "accuracy": 0.6,
                "macro_f1": 0.55,
                "min_class_accuracy": 0.3,
            },
            learning_rate=1e-4,
            epoch_duration_s=1.23,
        )

        tracker.finish()

        ok, missing = tracker.all_artifacts_present()
        assert ok, f"Missing or empty artifacts: {missing}"

    def test_run_directory_created_under_experiment_name(self, tmp_path):
        """run_dir must be under runs/{experiment_name}/."""
        tracker = ExperimentTracker(
            experiment_name="my_run",
            cfg=self._dummy_cfg(),
            base_dir=tmp_path / "runs",
        )
        tracker.start(seed=0)
        tracker.finish()
        assert (tmp_path / "runs" / "my_run").is_dir()
        assert tracker.run_dir.parent.name == "my_run"

    def test_config_yaml_matches_input_cfg(self, tmp_path):
        """config.yaml written by tracker must round-trip to the original cfg."""
        import yaml

        cfg = self._dummy_cfg()
        tracker = ExperimentTracker(
            experiment_name="cfg_test",
            cfg=cfg,
            base_dir=tmp_path / "runs",
        )
        tracker.start(seed=1)
        tracker.finish()

        with (tracker.run_dir / "config.yaml").open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        assert loaded["training"]["base_lr"] == cfg["training"]["base_lr"]
        assert loaded["data"]["image_size"] == cfg["data"]["image_size"]

    def test_metrics_csv_has_correct_columns_and_rows(self, tmp_path):
        """metrics.csv must have one header row + one data row per logged epoch."""
        import csv as csv_module

        cfg = self._dummy_cfg()
        tracker = ExperimentTracker(
            experiment_name="csv_test",
            cfg=cfg,
            base_dir=tmp_path / "runs",
        )
        tracker.start(seed=2)

        for ep in range(1, 4):
            tracker.log_epoch(
                epoch=ep,
                train_metrics={"loss": float(ep), "ce_loss": float(ep), "contrastive_loss": 0.0},
                val_metrics={
                    "loss": float(ep),
                    "accuracy": 0.5,
                    "macro_f1": 0.5,
                    "min_class_accuracy": 0.2,
                },
                learning_rate=1e-4,
                epoch_duration_s=0.1,
            )

        # Write dummy checkpoints so finish() doesn't error
        torch.save({}, tracker.best_checkpoint_path)
        torch.save({}, tracker.last_checkpoint_path)
        tracker.finish()

        with (tracker.run_dir / "metrics.csv").open("r", newline="", encoding="utf-8") as f:
            rows = list(csv_module.DictReader(f))

        assert len(rows) == 3
        expected_cols = {
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
        }
        assert expected_cols.issubset(set(rows[0].keys()))
        assert int(rows[0]["epoch"]) == 1
        assert int(rows[2]["epoch"]) == 3

    def test_seed_txt_contains_seed(self, tmp_path):
        """seed.txt must contain exactly the integer seed passed to start()."""
        tracker = ExperimentTracker(
            experiment_name="seed_test",
            cfg=self._dummy_cfg(),
            base_dir=tmp_path / "runs",
        )
        tracker.start(seed=1337)
        tracker.finish()

        content = (tracker.run_dir / "seed.txt").read_text(encoding="utf-8").strip()
        assert content == "1337"

    def test_runtime_txt_is_non_empty_after_finish(self, tmp_path):
        """runtime.txt must be written and non-empty after finish()."""
        tracker = ExperimentTracker(
            experiment_name="runtime_test",
            cfg=self._dummy_cfg(),
            base_dir=tmp_path / "runs",
        )
        tracker.start(seed=0)
        tracker.finish()

        content = (tracker.run_dir / "runtime.txt").read_text(encoding="utf-8").strip()
        assert content  # non-empty
        # Should parse as a float (seconds)
        assert float(content.rstrip("s")) >= 0.0

    def test_training_log_captures_logger_output(self, tmp_path):
        """training.log must exist and contain log output after a run."""
        cfg = self._dummy_cfg()
        tracker = ExperimentTracker(
            experiment_name="log_test",
            cfg=cfg,
            base_dir=tmp_path / "runs",
        )
        tracker.start(seed=5)
        # The start() call logs a line; log_epoch logs another
        tracker.log_epoch(
            epoch=1,
            train_metrics={"loss": 1.0, "ce_loss": 1.0, "contrastive_loss": 0.0},
            val_metrics={"loss": 0.9, "accuracy": 0.7, "macro_f1": 0.65, "min_class_accuracy": 0.4},
            learning_rate=1e-4,
        )
        tracker.finish()

        log_text = (tracker.run_dir / "training.log").read_text(encoding="utf-8")
        assert len(log_text) > 0
        assert "ExperimentTracker" in log_text or "Epoch" in log_text

    def test_context_manager_calls_finish(self, tmp_path):
        """Using ExperimentTracker as context manager must call finish()."""
        cfg = self._dummy_cfg()
        with ExperimentTracker(
            experiment_name="ctx_test",
            cfg=cfg,
            base_dir=tmp_path / "runs",
        ) as tracker:
            tracker.start(seed=7)

        # runtime.txt is only written in finish() — if it exists, finish was called
        assert (tracker.run_dir / "runtime.txt").exists()

    def test_tensorboard_event_file_created(self, tmp_path):
        """P6.6: At least one TensorBoard event file must be in the run_dir."""
        cfg = self._dummy_cfg()
        tracker = ExperimentTracker(
            experiment_name="tb_test",
            cfg=cfg,
            base_dir=tmp_path / "runs",
        )
        tracker.start(seed=0)
        tracker.log_epoch(
            epoch=1,
            train_metrics={"loss": 1.0, "ce_loss": 1.0, "contrastive_loss": 0.0},
            val_metrics={"loss": 0.9, "accuracy": 0.7, "macro_f1": 0.6, "min_class_accuracy": 0.3},
            learning_rate=1e-4,
        )
        tracker.finish()

        tb_events = list(tracker.run_dir.glob("events.out.tfevents.*"))
        assert len(tb_events) >= 1, "No TensorBoard event files found in run_dir"


# ===========================================================================
# Stage D — P6.8 (Trainer) & P6.9 (scripts/train.py) tests
# ===========================================================================

# ---------------------------------------------------------------------------
# Lightweight fake dataset + model for Trainer unit tests
# (avoids loading real CADDY data or EfficientNet-B4)
# ---------------------------------------------------------------------------


def _make_fake_csv(tmp_path: Path, n: int = 20, num_classes: int = 4) -> Path:
    """Write a minimal split CSV that CaddyStereoDataset can parse."""
    rows = []
    for i in range(n):
        rows.append(
            {
                "sample_id": f"s{i:04d}",
                "left_path": f"/fake/left_{i}.png",
                "right_path": f"/fake/right_{i}.png",
                "gesture_id": i % num_classes,
                "gesture_label": f"g{i % num_classes}",
                "scenario_id": i % 3,
            }
        )
    p = tmp_path / f"{uuid.uuid4().hex}_split.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def _make_tiny_cfg(num_classes: int = 4) -> dict:
    """Return a minimal config dict for Trainer tests."""
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
            "epochs": 1,
            "batch_size": 4,
            "optimizer": "adamw",
            "base_lr": 1e-3,
            "weight_decay": 1e-4,
            "scheduler": "cosine_annealing",
            "scheduler_eta_min_ratio": 0.01,
            "t_max": 1,
            "early_stopping_patience": 5,
            "loss": {"alpha": 1.0, "beta": 0.0, "class_weighted": True},
            "phases": {
                "phase1": {"epochs": [1, 1], "freeze_backbone": True, "lr_scale": 1.0},
                "phase2": {
                    "epochs": [2, 5],
                    "freeze_backbone": False,
                    "unfreeze_last_n_blocks": 2,
                    "lr_scale": 0.1,
                },
                "phase3": {
                    "epochs": [6, 10],
                    "freeze_backbone": False,
                    "unfreeze_last_n_blocks": None,
                    "lr_scale": 0.01,
                },
            },
        },
    }


class FakeStereoDataset(torch.utils.data.Dataset):
    """Tiny in-memory dataset that mimics CaddyStereoDataset output format."""

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


class TestTrainer:
    """P6.8 — Trainer unit tests (using patched dataloaders to avoid disk I/O)."""

    def _build_trainer(
        self, tmp_path: Path, num_classes: int = 4, cfg: dict | None = None
    ) -> Trainer:
        """Build a Trainer with fake dataloaders patched in."""
        from src.models.sense import SENSE

        if cfg is None:
            cfg = _make_tiny_cfg(num_classes)
        train_csv = _make_fake_csv(tmp_path, n=20, num_classes=num_classes)
        val_csv = _make_fake_csv(tmp_path, n=8, num_classes=num_classes)
        model = SENSE.from_config(cfg)

        # Patch get_dataloader so Trainer never touches disk images
        fake_loader = _make_fake_loader(n=20, num_classes=num_classes, batch_size=4)
        fake_val_loader = _make_fake_loader(n=8, num_classes=num_classes, batch_size=4)

        with patch(
            "src.training.trainer.get_dataloader", side_effect=[fake_loader, fake_val_loader]
        ):
            trainer = Trainer(
                model=model,
                cfg=cfg,
                train_csv=str(train_csv),
                val_csv=str(val_csv),
                experiment_name="test_run",
                device="cpu",
            )
        # Swap in persistent fake loaders (the patched ones are consumed during __init__)
        trainer.train_loader = _make_fake_loader(n=20, num_classes=num_classes)
        trainer.val_loader = _make_fake_loader(n=8, num_classes=num_classes)
        return trainer

    def test_trainer_single_epoch_loss_finite_and_checkpoint_saved(self, tmp_path):
        """
        P6.8 Verification: one epoch on a tiny subset completes,
        logs correctly, saves a checkpoint, loss is finite.
        """
        trainer = self._build_trainer(tmp_path)
        best = trainer.fit(num_epochs=1)

        assert "macro_f1" in best
        assert "accuracy" in best
        assert "min_class_accuracy" in best
        for v in best.values():
            assert isinstance(v, float)
            assert not (v != v)  # not NaN

        # Checkpoint files must exist
        assert trainer.tracker.last_checkpoint_path.exists()
        assert trainer.tracker.best_checkpoint_path.exists()

    def test_trainer_fit_never_opens_test_split(self, tmp_path):
        """
        Required test: fit() must not open/load any test-split CSV.
        We track every pd.read_csv call and assert that none matches a test-split path.
        """
        trainer = self._build_trainer(tmp_path)
        opened_paths: list[str] = []

        real_read_csv = pd.read_csv

        def tracked_read_csv(path_or_buf, *args, **kwargs):
            if isinstance(path_or_buf, (str, Path)):
                opened_paths.append(str(path_or_buf))
            return real_read_csv(path_or_buf, *args, **kwargs)

        with patch("pandas.read_csv", side_effect=tracked_read_csv):
            trainer.fit(num_epochs=1)

        # Check that no opened path contains "test" in a CSV filename
        test_paths = [p for p in opened_paths if "test" in Path(p).stem.lower()]
        assert len(test_paths) == 0, f"fit() accessed test-split paths: {test_paths}"

    def test_trainer_class_weights_from_training_csv(self, tmp_path):
        """
        Required test: the weights used by the loss are computed from
        that run's training CSV (not hardcoded or from val CSV).
        """
        trainer = self._build_trainer(tmp_path)

        # SenseLoss should have class_weights populated from train_csv
        # (because _make_tiny_cfg sets class_weighted: True)
        assert (
            trainer.loss_fn.class_weights is not None
        ), "class_weights should be non-None when class_weighted=True"
        # Shape: (num_classes,)
        assert trainer.loss_fn.class_weights.shape == (4,)

    def test_validate_produces_finite_metrics(self, tmp_path):
        """validate() against val loader returns finite metric dict."""
        trainer = self._build_trainer(tmp_path)
        metrics = trainer.validate()
        for key in ("loss", "accuracy", "macro_f1", "min_class_accuracy"):
            assert key in metrics
            assert isinstance(metrics[key], float)
            assert not (metrics[key] != metrics[key])  # not NaN

    def test_trainer_all_artifacts_present_after_fit(self, tmp_path):
        """All nine ExperimentTracker artifacts must be present after fit()."""
        trainer = self._build_trainer(tmp_path)
        trainer.fit(num_epochs=1)

        ok, missing = trainer.tracker.all_artifacts_present()
        assert ok, f"Missing or empty artifacts after fit(): {missing}"


class _ForceNaNLoss:
    """
    Callable wrapper around SenseLoss that can inject non-finite losses.

    Used only by divergence-guard tests — never mutates the wrapped loss.
    Mirrors the Trainer's calling convention for SenseLoss:
      - train steps: ``loss_fn(logits, labels, return_components=True)``
      - val steps:   ``loss_fn(logits, labels)``
    """

    def __init__(self, inner: SenseLoss) -> None:
        self.inner = inner
        # NaN from the (N+1)-th train call onward (N finite calls first).
        self.nan_after_train_calls: int | None = None
        # Single NaN at the N-th train call (1-indexed).
        self.nan_train_call: int | None = None
        # NaN for every validation call.
        self.nan_val: bool = False
        self._train_calls = 0

    def __call__(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        text_embeddings: torch.Tensor | None = None,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        force_nan = False
        if return_components:
            self._train_calls += 1
            if (
                self.nan_after_train_calls is not None
                and self._train_calls > self.nan_after_train_calls
            ):
                force_nan = True
            if self.nan_train_call is not None and self._train_calls == self.nan_train_call:
                force_nan = True
        elif self.nan_val:
            force_nan = True

        if force_nan:
            nan = torch.tensor(float("nan"), device=logits.device)
            if return_components:
                components = {
                    "loss": float("nan"),
                    "ce_loss": float("nan"),
                    "contrastive_loss": float("nan"),
                }
                return nan, components
            return nan

        return self.inner(
            logits,
            targets,
            text_embeddings=text_embeddings,
            return_components=return_components,
        )


class TestDivergenceGuard:
    """
    Stage E — fail-fast NaN/inf divergence guard in Trainer.fit().

    (a) 20 consecutive non-finite training losses abort the run.
    (b) A single isolated non-finite training step does not abort.
    (c) A non-finite validation loss aborts BEFORE the end-of-epoch
        last.pth save, leaving the previous healthy checkpoint loadable.
    """

    def _build_trainer(self, tmp_path: Path) -> Trainer:
        return TestTrainer()._build_trainer(tmp_path)

    def test_consecutive_nan_train_loss_aborts_and_best_untouched(self, tmp_path):
        """
        (a) A loss forced to NaN aborts within 20 consecutive steps, and
        best.pth still holds the last healthy (epoch-1) checkpoint.
        """
        trainer = self._build_trainer(tmp_path)
        wrapper = _ForceNaNLoss(trainer.loss_fn)
        # 5 train batches per epoch: epoch 1 finite, NaN from epoch 2 onward.
        wrapper.nan_after_train_calls = 5
        trainer.loss_fn = wrapper

        with pytest.raises(RuntimeError) as excinfo:
            trainer.fit(num_epochs=5)
        msg = str(excinfo.value)
        assert "epoch" in msg.lower(), f"Abort message lacks epoch info: {msg}"
        assert "steps" in msg.lower(), f"Abort message lacks step info: {msg}"
        assert trainer._consecutive_nonfinite_steps >= trainer.nan_step_tolerance

        # best.pth must still be the epoch-1 checkpoint with finite metrics.
        payload = load_checkpoint(trainer.tracker.best_checkpoint_path)
        assert int(payload["epoch"]) == 1
        assert all(math.isfinite(float(v)) for v in payload["metrics"].values())

    def test_single_nonfinite_step_does_not_abort(self, tmp_path):
        """
        (b) One isolated non-finite training step is tolerated: the run
        completes, the poisoned update is skipped (no backward), and the
        consecutive counter resets to zero on the following finite steps.
        """
        trainer = self._build_trainer(tmp_path)
        wrapper = _ForceNaNLoss(trainer.loss_fn)
        wrapper.nan_train_call = 2  # second train batch of epoch 1 only
        trainer.loss_fn = wrapper

        trainer.fit(num_epochs=1)  # must not raise

        assert trainer.tracker.last_checkpoint_path.exists()
        assert trainer._consecutive_nonfinite_steps == 0
        payload = load_checkpoint(trainer.tracker.last_checkpoint_path)
        assert all(math.isfinite(float(v)) for v in payload["metrics"].values())

    def test_nonfinite_val_loss_aborts_before_last_checkpoint_save(self, tmp_path):
        """
        (c) A non-finite validation loss aborts BEFORE the end-of-epoch
        last.pth save, so last.pth still holds the previous healthy epoch
        and remains loadable (resume-ability).
        """
        trainer = self._build_trainer(tmp_path)
        trainer.fit(num_epochs=2)  # epochs 1-2 healthy; last.pth at epoch 2
        last_path = trainer.tracker.last_checkpoint_path
        assert int(load_checkpoint(last_path)["epoch"]) == 2

        wrapper = _ForceNaNLoss(trainer.loss_fn)
        wrapper.nan_val = True
        trainer.loss_fn = wrapper

        with pytest.raises(RuntimeError) as excinfo:
            trainer.fit(num_epochs=1)
        assert "epoch" in str(excinfo.value).lower()

        # last.pth must still be the epoch-2 checkpoint (poisoned epoch never saved)
        payload_after = load_checkpoint(last_path)
        assert int(payload_after["epoch"]) == 2
        # and it remains loadable into the model
        load_checkpoint(last_path, model=trainer.model)


class TestAMPOrderingGuard:
    """
    Regression guard (audit item 1): the mixed-precision step order in
    Trainer._train_epoch must remain

        scaler.scale(loss).backward()
        -> scaler.unscale_(optimizer)
        -> clip_grad_norm_(max_norm=1.0)
        -> scaler.step(optimizer)
        -> scaler.update()

    unscale_ MUST precede clip_grad_norm_ (clipping must measure true unscaled
    gradients) and scaler.step (GradScaler skips steps whose gradients contain
    inf/NaN, which it detects during unscale_).
    """

    def _build_trainer(self, tmp_path: Path) -> Trainer:
        return TestTrainer()._build_trainer(tmp_path)

    def test_amp_call_order_is_scale_backward_unscale_clip_step_update(self, tmp_path):
        trainer = self._build_trainer(tmp_path)

        calls: list[str] = []

        class _ScaledLossProxy:
            def __init__(self, loss: torch.Tensor) -> None:
                self._loss = loss

            def backward(self) -> None:
                calls.append("backward")
                self._loss.backward()

        class _RecordingScaler:
            """Records scaler call order and delegates to the real scaler."""

            def __init__(self, real) -> None:
                self._real = real

            def scale(self, loss: torch.Tensor) -> _ScaledLossProxy:
                calls.append("scale")
                return _ScaledLossProxy(loss)

            def unscale_(self, optimizer) -> None:
                calls.append("unscale_")
                self._real.unscale_(optimizer)

            def step(self, optimizer) -> None:
                calls.append("step")
                self._real.step(optimizer)

            def update(self) -> None:
                calls.append("update")
                self._real.update()

        real_clip = torch.nn.utils.clip_grad_norm_

        def recording_clip(params, max_norm, **kwargs):
            calls.append("clip")
            return real_clip(params, max_norm=max_norm, **kwargs)

        trainer.scaler = _RecordingScaler(trainer.scaler)

        expected_cycle = ["scale", "backward", "unscale_", "clip", "step", "update"]

        with patch("torch.nn.utils.clip_grad_norm_", side_effect=recording_clip):
            # 20 fake samples / batch size 4 = 5 batches -> 5 identical cycles
            trainer._train_epoch(epoch=1)

        assert calls[:6] == expected_cycle, f"AMP step order wrong: {calls[:6]}"
        assert len(calls) == 30, f"Expected 5 complete AMP cycles, got {len(calls)} calls"
        # Every cycle must keep the correct relative order.
        for start in range(0, len(calls), 6):
            assert calls[start : start + 6] == expected_cycle, f"Cycle at {start} wrong: {calls}"


class TestNumericsHardening:
    """
    Audit item 2 — fp32 hardening of the cosine head and the losses.

    Inside an active autocast region the head's matmul would otherwise be
    downcast (fp16 on CUDA / bf16 on CPU); the hardened implementation must
    return fp32 regardless, so overflow safety no longer depends on the
    autocast op list.  CPU bf16 autocast is used so these tests run in the
    plain (not-slow) CPU test suite.
    """

    def test_cosine_head_returns_fp32_under_autocast(self):
        from src.models.classifier_head import CosineSimilarityHead

        head = CosineSimilarityHead(embed_dim=16, num_classes=5)
        visual = torch.randn(4, 16)
        class_emb = torch.randn(5, 16)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = head(visual, class_emb)
        assert out.dtype == torch.float32
        assert out.shape == (4, 5)

    def test_cosine_head_batched_embeddings_fp32_under_autocast(self):
        from src.models.classifier_head import CosineSimilarityHead

        head = CosineSimilarityHead(embed_dim=16, num_classes=5)
        visual = torch.randn(2, 16)
        class_emb = torch.randn(2, 5, 16)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = head(visual, class_emb)
        assert out.dtype == torch.float32
        assert out.shape == (2, 5)

    def test_contrastive_loss_returns_fp32_under_autocast(self):
        loss_fn = ContrastiveLoss(temperature=0.07)
        logits = torch.randn(4, 5)
        targets = torch.randint(0, 5, (4,))
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = loss_fn(logits, targets)
        assert out.dtype == torch.float32

    def test_sense_loss_ce_returns_fp32_under_autocast(self):
        loss_fn = SenseLoss(alpha=1.0, beta=0.0, text_enabled=False)
        logits = torch.randn(4, 5)
        targets = torch.randint(0, 5, (4,))
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = loss_fn(logits, targets)
        assert out.dtype == torch.float32


class TestGradientNormLogging:
    """
    Audit item 4 — per-epoch mean unscaled gradient norm and clipping-active
    fraction must be logged to training.log (via the module logger) and stored
    on the trainer, never written to metrics.csv.
    """

    def _build_trainer(self, tmp_path: Path) -> Trainer:
        return TestTrainer()._build_trainer(tmp_path)

    def test_epoch_logs_grad_norm_summary_and_stats(self, tmp_path, caplog):
        import logging as _logging

        trainer = self._build_trainer(tmp_path)

        with caplog.at_level(_logging.INFO, logger="src.training.trainer"):
            trainer._train_epoch(epoch=1)

        messages = [
            record.getMessage()
            for record in caplog.records
            if "mean unscaled grad norm" in record.getMessage()
        ]
        assert messages, "Epoch gradient summary was not logged"
        assert "clipping active in" in messages[0]

        stats = trainer._last_grad_stats
        assert stats["steps"] == 5  # 20 fake samples / batch size 4
        assert 0.0 <= stats["clip_fraction"] <= 1.0
        assert math.isfinite(stats["mean_norm"])
        assert stats["clipped"] <= stats["steps"]
        assert stats["skipped_batches"] == 0  # no NaN injected in this run


class TestEarlyStoppingPhaseAwareness:
    """Pre-flight check C1: Early stopping must not kill run during frozen phase."""

    def test_early_stopping_guards_frozen_phase_and_resets_counter(self, tmp_path):
        """
        Simulate flat validation metrics from epoch 1.
        Verify that training continues past epoch 10 and that the early stopping
        counter resets at the phase transition (epoch 11).
        """
        cfg = _make_tiny_cfg()
        cfg["training"]["epochs"] = 20
        cfg["training"]["t_max"] = 20
        cfg["training"]["phases"] = {
            "phase1": {"epochs": [1, 10], "freeze_backbone": True, "lr_scale": 1.0},
            "phase2": {
                "epochs": [11, 15],
                "freeze_backbone": False,
                "unfreeze_last_n_blocks": 2,
                "lr_scale": 0.1,
            },
            "phase3": {
                "epochs": [16, 20],
                "freeze_backbone": False,
                "unfreeze_last_n_blocks": None,
                "lr_scale": 0.01,
            },
        }
        trainer_holder = TestTrainer()
        trainer = trainer_holder._build_trainer(tmp_path, cfg=cfg)
        trainer.patience = 10
        assert trainer._first_unfreeze_epoch == 11

        trainer._train_epoch = MagicMock(
            return_value={"loss": 1.0, "ce_loss": 1.0, "contrastive_loss": 0.0}
        )
        trainer._validate_epoch = MagicMock(
            return_value={
                "loss": 1.0,
                "accuracy": 0.50,
                "macro_f1": 0.40,
                "min_class_accuracy": 0.20,
            }
        )

        epochs_visited: list[int] = []
        counter_before_val: list[int] = []

        orig_train_epoch = trainer._train_epoch

        def tracked_train_epoch(epoch):
            epochs_visited.append(epoch)
            counter_before_val.append(trainer._epochs_no_improve)
            return orig_train_epoch(epoch)

        trainer._train_epoch = tracked_train_epoch

        # Run 12 epochs: epoch 1 sets best (counter 0). Epochs 2-10 increment counter to 9.
        # At start of epoch 11, phase transition fires and resets counter to 0!
        trainer.fit(num_epochs=12)

        # Confirm epoch 11 and 12 ran (past frozen phase epoch 10)
        assert 11 in epochs_visited
        assert 12 in epochs_visited
        # Confirm that at start of epoch 11 (after transition), counter was reset to 0
        idx_11 = epochs_visited.index(11)
        assert (
            counter_before_val[idx_11] == 0
        ), f"Counter at epoch 11 start was {counter_before_val[idx_11]}, expected 0"

    def test_early_stopping_fires_after_first_unfreeze_when_patience_exceeded(self, tmp_path):
        """
        When validation metric never improves after epoch 11, early stopping
        must fire once patience (10) epochs elapse without improvement (at epoch 21).
        """
        cfg = _make_tiny_cfg()
        cfg["training"]["epochs"] = 30
        cfg["training"]["t_max"] = 30
        cfg["training"]["phases"] = {
            "phase1": {"epochs": [1, 10], "freeze_backbone": True, "lr_scale": 1.0},
            "phase2": {
                "epochs": [11, 25],
                "freeze_backbone": False,
                "unfreeze_last_n_blocks": 2,
                "lr_scale": 0.1,
            },
            "phase3": {
                "epochs": [26, 30],
                "freeze_backbone": False,
                "unfreeze_last_n_blocks": None,
                "lr_scale": 0.01,
            },
        }
        trainer_holder = TestTrainer()
        trainer = trainer_holder._build_trainer(tmp_path, cfg=cfg)
        trainer.patience = 10

        trainer._train_epoch = MagicMock(
            return_value={"loss": 1.0, "ce_loss": 1.0, "contrastive_loss": 0.0}
        )
        trainer._validate_epoch = MagicMock(
            return_value={
                "loss": 1.0,
                "accuracy": 0.50,
                "macro_f1": 0.40,
                "min_class_accuracy": 0.20,
            }
        )

        epochs_visited: list[int] = []
        orig_train_epoch = trainer._train_epoch

        def tracked_train_epoch(epoch):
            epochs_visited.append(epoch)
            return orig_train_epoch(epoch)

        trainer._train_epoch = tracked_train_epoch

        # Run up to 30 epochs with flat metrics.
        # Phase 1: epochs 1-10 (no stop because epoch < 11).
        # Epoch 11: transition resets counter to 0, then after validation counter becomes 1.
        # Epochs 11-20: 10 consecutive non-improving epochs (counters 1..10).
        # At epoch 20 end: _epochs_no_improve reaches 10 >= patience (and 20 >= 11), triggers early stop!
        trainer.fit(num_epochs=30)
        assert (
            max(epochs_visited) == 20
        ), f"Expected early stop at epoch 20, got {max(epochs_visited)}"


class TestProgressiveLRSchedule:
    """Pre-flight check C3: LR schedule and unfreezing phases."""

    def test_lr_schedule_across_unfreezing_phases(self, tmp_path):
        """
        Dry-run through unfreezing phases without data.
        Confirms cosine decay of base_lr, lr_scale proportionality for backbone,
        dynamic base_lrs synchronization, and actual updates to newly unfrozen parameters.
        """
        from src.models.sense import SENSE

        cfg = _make_tiny_cfg()
        cfg["training"]["base_lr"] = 1e-4
        cfg["training"]["epochs"] = 50
        cfg["training"]["t_max"] = 50
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
        cfg["data"]["split_train"] = str(tmp_path / "train.csv")
        cfg["data"]["split_val"] = str(tmp_path / "val.csv")

        model = SENSE.from_config(cfg)
        opt = build_optimizer(model, cfg)
        sched = build_scheduler(opt, cfg)
        ctrl = PhaseController(model=model, cfg=cfg, optimizer=opt, scheduler=sched)

        for epoch in range(1, 51):
            ctrl.step(epoch)

            # Check group count matches base_lrs
            assert len(sched.base_lrs) == len(opt.param_groups)

            head_lr = opt.param_groups[0]["lr"]
            s = epoch - 1
            cos_fac = 0.5 * (1.0 + math.cos(math.pi * s / 50))
            ratio = float(cfg["training"]["scheduler_eta_min_ratio"])
            expected_factor = ratio + (1.0 - ratio) * cos_fac
            assert math.isclose(head_lr, 1e-4 * expected_factor, rel_tol=1e-5)

            if len(opt.param_groups) > 1:
                scale = 0.1 if epoch <= 30 else 0.01
                backbone_lr = opt.param_groups[1]["lr"]
                assert math.isclose(backbone_lr, head_lr * scale, rel_tol=1e-5)

            # Check that unfrozen parameters receive updates
            if epoch in (11, 31):
                p_unfrozen = next(
                    p for p in model.encoder.left_branch[0].parameters() if p.requires_grad
                )
                before = p_unfrozen.clone().detach()
                p_unfrozen.grad = torch.ones_like(p_unfrozen) * 0.01
                opt.step()
                opt.zero_grad(set_to_none=True)
                after = p_unfrozen.clone().detach()
                assert not torch.equal(before, after)
            else:
                opt.step()

            sched.step()


class TestTrainScriptSmoke:
    """
    Stage D smoke test — P6.9 Verification:
    Run scripts/train.py --config configs/sense_full.yaml --epochs 1
    and confirm all nine experiment-record artifacts are produced.

    This test requires the actual CADDY data CSVs and enhanced images.
    It is marked 'slow' and skipped when data is unavailable.
    """

    @pytest.mark.slow
    def test_train_script_1epoch_smoke(self, tmp_path):
        """
        Run scripts/train.py for 1 epoch and verify the experiment record.
        Requires real data to be present (skipped otherwise).
        """
        project_root = Path(__file__).resolve().parent.parent
        train_csv = project_root / "data" / "processed" / "splits" / "standard_train.csv"
        val_csv = project_root / "data" / "processed" / "splits" / "standard_val.csv"
        config = project_root / "configs" / "sense_full.yaml"

        if not train_csv.exists() or not val_csv.exists():
            pytest.skip("Real CADDY split CSVs not available; skipping smoke test.")
        if not config.exists():
            pytest.skip("configs/sense_full.yaml not found.")

        exp_name = f"smoke_test_{uuid.uuid4().hex[:8]}"
        cmd = [
            sys.executable,
            str(project_root / "scripts" / "train.py"),
            "--config",
            str(config),
            "--epochs",
            "1",
            "--experiment-name",
            exp_name,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(project_root),
            timeout=600,  # 10-minute safety cap
        )

        if result.returncode != 0:
            pytest.fail(
                f"train.py exited with code {result.returncode}.\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )

        # Locate the most recently created run dir for this experiment
        runs_root = project_root / "runs" / exp_name
        assert runs_root.exists(), f"Run directory not found: {runs_root}"
        run_dirs = sorted(runs_root.iterdir(), key=lambda d: d.stat().st_mtime)
        assert run_dirs, f"No run sub-directories found under {runs_root}"
        run_dir = run_dirs[-1]  # most recent

        required_artifacts = [
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
        missing = []
        for rel in required_artifacts:
            p = run_dir / rel
            if not p.exists():
                missing.append(f"{rel} (missing)")
            elif rel != "gpu_info.txt" and p.stat().st_size == 0:
                missing.append(f"{rel} (empty)")

        # TensorBoard event file
        tb_events = list(run_dir.glob("events.out.tfevents.*"))
        if not tb_events:
            missing.append("TensorBoard event file (missing)")

        assert not missing, f"Smoke test: missing/empty artifacts in {run_dir}:\n" + "\n".join(
            missing
        )
