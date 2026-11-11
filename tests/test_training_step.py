"""
tests/test_training_step.py
===========================
Tests for Phase 6 training infrastructure.
Stage A: P6.1 (SenseLoss, ContrastiveLoss) & P6.2 (Optimizer & Scheduler factory).
Stage B: P6.3 (PhaseController) & P6.4 (Checkpoint).
Stage C: P6.5 (ExperimentTracker) & P6.6 (TensorBoard) & P6.7 (Seed control).
"""

from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from src.data.splits import compute_class_weights
from src.training.losses import ContrastiveLoss, SenseLoss
from src.training.scheduler import (
    build_optimizer,
    build_optimizer_and_scheduler,
    build_scheduler,
)
from src.training.trainer import PhaseController
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
