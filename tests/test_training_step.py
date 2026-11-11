"""
tests/test_training_step.py
===========================
Tests for Phase 6 training infrastructure.
Stage A: P6.1 (SenseLoss, ContrastiveLoss) & P6.2 (Optimizer & Scheduler factory).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn as nn

from src.data.splits import compute_class_weights
from src.training.losses import ContrastiveLoss, SenseLoss
from src.training.scheduler import (
    build_optimizer,
    build_optimizer_and_scheduler,
    build_scheduler,
)


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
