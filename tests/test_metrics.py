"""
tests/test_metrics.py
=====================
Unit tests for src/evaluation/metrics.py (Stage 0 / P8.1).

Verifies all core metric functions on hand-computed synthetic examples,
tie-breaker logic, latency measurement, and matrix visualization.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics import (
    compute_confusion_matrix,
    compute_macro_f1,
    compute_min_class_accuracy,
    compute_overall_accuracy,
    compute_per_class_accuracy,
    compute_precision_recall_f1_per_class,
    evaluate_predictions,
    is_better_checkpoint,
    measure_inference_time,
    plot_confusion_matrix,
)


class TestOverallAccuracy:
    def test_hand_computed_exact(self):
        # 6 samples: 5 correct, 1 wrong -> 5 / 6 = 0.8333333333333334
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 1, 0, 1, 2]
        acc = compute_overall_accuracy(y_true, y_pred)
        assert pytest.approx(acc, abs=1e-6) == 5.0 / 6.0

    def test_with_torch_tensors(self):
        y_true = torch.tensor([0, 1, 2, 3])
        y_pred = torch.tensor([0, 1, 2, 0])
        acc = compute_overall_accuracy(y_true, y_pred)
        assert pytest.approx(acc, abs=1e-6) == 0.75

    def test_with_2d_logits(self):
        # Logits of shape (4, 3)
        y_true = np.array([0, 1, 2, 1])
        logits = np.array(
            [
                [5.0, 1.0, 0.0],  # argmax 0 -> correct
                [0.0, 4.0, 1.0],  # argmax 1 -> correct
                [1.0, 2.0, 0.0],  # argmax 1 -> wrong (true 2)
                [0.5, 3.0, 1.0],  # argmax 1 -> correct
            ]
        )
        acc = compute_overall_accuracy(y_true, logits)
        assert pytest.approx(acc, abs=1e-6) == 0.75

    def test_empty_returns_zero(self):
        assert compute_overall_accuracy([], []) == 0.0


class TestMacroF1:
    def test_hand_computed_3_class(self):
        """
        Hand calculation for 3 classes:
          y_true = [0, 0, 1, 1, 2, 2]
          y_pred = [0, 1, 1, 1, 2, 0]

          Class 0:
            TP = 1 (idx 0)
            FP = 1 (idx 5 pred 0, true 2)
            FN = 1 (idx 1 pred 1, true 0)
            Prec = 1/2 = 0.5, Rec = 1/2 = 0.5, F1_0 = 0.5

          Class 1:
            TP = 2 (idx 2, 3)
            FP = 1 (idx 1 pred 1, true 0)
            FN = 0
            Prec = 2/3, Rec = 2/2 = 1.0, F1_1 = 2*(2/3 * 1) / (2/3 + 1) = (4/3) / (5/3) = 0.8

          Class 2:
            TP = 1 (idx 4)
            FP = 0
            FN = 1 (idx 5 pred 0, true 2)
            Prec = 1/1 = 1.0, Rec = 1/2 = 0.5, F1_2 = 2*(1 * 0.5) / (1 + 0.5) = 1.0 / 1.5 = 2/3

          Macro F1 = (0.5 + 0.8 + 2/3) / 3 = 1.9666666667 / 3 = 0.6555555556
        """
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 1, 1, 1, 2, 0]

        expected_macro_f1 = (0.5 + 0.8 + (2.0 / 3.0)) / 3.0
        f1 = compute_macro_f1(y_true, y_pred, num_classes=3)
        assert pytest.approx(f1, abs=1e-6) == expected_macro_f1

    def test_unrepresented_class_penalized(self):
        # 4 classes, but class 3 is absent in predictions and truth
        # num_classes=4 forces class 3 with F1=0.0 into the macro average
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 0, 1, 1, 2, 2]
        # F1 for classes 0, 1, 2 is 1.0 each; class 3 has F1 = 0.0
        # Macro F1 should be (1.0 + 1.0 + 1.0 + 0.0) / 4 = 0.75
        f1_4 = compute_macro_f1(y_true, y_pred, num_classes=4)
        assert pytest.approx(f1_4, abs=1e-6) == 0.75

        # Without num_classes=4, only 0, 1, 2 are present -> 1.0
        f1_3 = compute_macro_f1(y_true, y_pred, num_classes=None)
        assert pytest.approx(f1_3, abs=1e-6) == 1.0


class TestPerClassAccuracyAndMin:
    def test_hand_computed_accuracies(self):
        """
        y_true = [0, 0, 0, 1, 1, 2, 2, 2, 2]
        y_pred = [0, 0, 1, 1, 0, 2, 2, 2, 1]

        Class 0: 3 total, 2 correct -> 2/3 = 0.666667
        Class 1: 2 total, 1 correct -> 1/2 = 0.500000
        Class 2: 4 total, 3 correct -> 3/4 = 0.750000
        Minimum class accuracy = 0.500000 (Class 1)
        """
        y_true = [0, 0, 0, 1, 1, 2, 2, 2, 2]
        y_pred = [0, 0, 1, 1, 0, 2, 2, 2, 1]

        per_class = compute_per_class_accuracy(y_true, y_pred, num_classes=3)
        assert pytest.approx(per_class[0], abs=1e-6) == 2.0 / 3.0
        assert pytest.approx(per_class[1], abs=1e-6) == 0.5
        assert pytest.approx(per_class[2], abs=1e-6) == 0.75

        min_acc = compute_min_class_accuracy(y_true, y_pred, num_classes=3)
        assert pytest.approx(min_acc, abs=1e-6) == 0.5

    def test_zero_sample_class(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]

        # Class 2 has 0 samples:
        per_class = compute_per_class_accuracy(y_true, y_pred, num_classes=3)
        assert per_class[2] == 0.0

        # with only_present=True, minimum over {0, 1} is 1.0
        min_acc_present = compute_min_class_accuracy(
            y_true, y_pred, num_classes=3, only_present=True
        )
        assert min_acc_present == 1.0

        # with only_present=False, class 2 with 0.0 is included -> 0.0
        min_acc_all = compute_min_class_accuracy(y_true, y_pred, num_classes=3, only_present=False)
        assert min_acc_all == 0.0


class TestPrecisionRecallF1PerClass:
    def test_matches_hand_computed(self):
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 1, 1, 1, 2, 0]

        res = compute_precision_recall_f1_per_class(y_true, y_pred, num_classes=3)
        # Class 0: Prec 0.5, Rec 0.5, F1 0.5, support 2
        assert pytest.approx(res[0]["precision"], abs=1e-6) == 0.5
        assert pytest.approx(res[0]["recall"], abs=1e-6) == 0.5
        assert pytest.approx(res[0]["f1"], abs=1e-6) == 0.5
        assert res[0]["support"] == 2

        # Class 1: Prec 2/3, Rec 1.0, F1 0.8, support 2
        assert pytest.approx(res[1]["precision"], abs=1e-6) == 2.0 / 3.0
        assert pytest.approx(res[1]["recall"], abs=1e-6) == 1.0
        assert pytest.approx(res[1]["f1"], abs=1e-6) == 0.8
        assert res[1]["support"] == 2

        # Class 2: Prec 1.0, Rec 0.5, F1 2/3, support 2
        assert pytest.approx(res[2]["precision"], abs=1e-6) == 1.0
        assert pytest.approx(res[2]["recall"], abs=1e-6) == 0.5
        assert pytest.approx(res[2]["f1"], abs=1e-6) == 2.0 / 3.0
        assert res[2]["support"] == 2


class TestConfusionMatrix:
    def test_matrix_entries(self):
        y_true = [0, 0, 0, 1, 1, 2, 2, 2, 2]
        y_pred = [0, 0, 1, 1, 0, 2, 2, 2, 1]

        cm = compute_confusion_matrix(y_true, y_pred, num_classes=3)
        expected = np.array(
            [
                [2, 1, 0],
                [1, 1, 0],
                [0, 1, 3],
            ]
        )
        np.testing.assert_array_equal(cm, expected)

    def test_plot_confusion_matrix_saves_file(self):
        cm = np.array([[10, 2], [1, 15]])
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "cm.png"
            fig = plot_confusion_matrix(
                cm,
                class_names=["class_a", "class_b"],
                save_path=save_path,
                normalize=True,
            )
            assert fig is not None
            assert save_path.exists()
            assert save_path.stat().st_size > 0


class TestTieBreakingRule:
    """
    Tests binding tie-breaking logic from the Evaluation Strategy:
    If within 0.5 macro-F1 points (0.005), tie is broken by minimum class accuracy.
    """

    def test_first_checkpoint_always_better(self):
        assert is_better_checkpoint(0.50, 0.40, None, None) is True

    def test_f1_clearly_better(self):
        # Current F1 is better by 0.010 (> 0.005 threshold) -> wins even with lower min_acc
        assert (
            is_better_checkpoint(
                val_macro_f1=0.850,
                val_min_acc=0.600,
                best_macro_f1=0.840,
                best_min_acc=0.700,
                f1_tolerance=0.005,
            )
            is True
        )

    def test_f1_clearly_worse(self):
        # Current F1 is worse by 0.010 (> 0.005 threshold) -> loses even with higher min_acc
        assert (
            is_better_checkpoint(
                val_macro_f1=0.830,
                val_min_acc=0.800,
                best_macro_f1=0.840,
                best_min_acc=0.700,
                f1_tolerance=0.005,
            )
            is False
        )

    def test_tie_f1_won_by_min_acc(self):
        # Current F1 is within 0.005 (0.842 vs 0.840, delta = +0.002)
        # Higher min_acc (0.750 vs 0.700) -> wins!
        assert (
            is_better_checkpoint(
                val_macro_f1=0.842,
                val_min_acc=0.750,
                best_macro_f1=0.840,
                best_min_acc=0.700,
                f1_tolerance=0.005,
            )
            is True
        )

    def test_tie_f1_lost_by_min_acc(self):
        # Current F1 is slightly higher by 0.002 (within 0.005)
        # But lower min_acc (0.650 vs 0.700) -> loses tie-break!
        assert (
            is_better_checkpoint(
                val_macro_f1=0.842,
                val_min_acc=0.650,
                best_macro_f1=0.840,
                best_min_acc=0.700,
                f1_tolerance=0.005,
            )
            is False
        )

    def test_percentage_format_scaling(self):
        # Both in 0..100 scale: 84.2 vs 84.0 (delta = 0.2 < 0.5)
        # Higher min_acc wins
        assert (
            is_better_checkpoint(
                val_macro_f1=84.2,
                val_min_acc=75.0,
                best_macro_f1=84.0,
                best_min_acc=70.0,
            )
            is True
        )


class TestInferenceTimeMeasurement:
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 2)

        def forward(self, left, right=None):
            return self.linear(left)

    def test_latency_measurement_positive(self):
        model = self.DummyModel()
        x = torch.randn(20, 10)
        ds = TensorDataset(x)
        loader = DataLoader(ds, batch_size=4)

        latency = measure_inference_time(
            model=model,
            dataloader=loader,
            device="cpu",
            num_batches=5,
            warmup_batches=1,
        )
        assert isinstance(latency, float)
        assert latency > 0.0

    def test_model_training_mode_restored(self):
        model = self.DummyModel()
        model.train()
        x = torch.randn(10, 10)
        loader = DataLoader(TensorDataset(x), batch_size=2)

        _ = measure_inference_time(model, loader, device="cpu", num_batches=2, warmup_batches=0)
        assert model.training is True


class TestEvaluatePredictionsBundle:
    def test_full_bundle_keys(self):
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 1, 1]

        bundle = evaluate_predictions(y_true, y_pred, num_classes=3)
        expected_keys = {
            "overall_accuracy",
            "macro_f1",
            "min_class_accuracy",
            "per_class_accuracy",
            "per_class_metrics",
            "confusion_matrix",
        }
        assert set(bundle.keys()) == expected_keys
        assert 0.0 <= bundle["overall_accuracy"] <= 1.0
        assert 0.0 <= bundle["macro_f1"] <= 1.0
        assert 0.0 <= bundle["min_class_accuracy"] <= 1.0
        assert isinstance(bundle["confusion_matrix"], np.ndarray)


class TestConfigIndependence:
    """Verifies arbitrary class counts work without hardcoding 17."""

    def test_caddy_17_classes(self):
        num_classes = 17
        y_true = list(range(num_classes))
        y_pred = list(range(num_classes))

        res = evaluate_predictions(y_true, y_pred, num_classes=num_classes)
        assert res["overall_accuracy"] == 1.0
        assert res["macro_f1"] == 1.0
        assert res["min_class_accuracy"] == 1.0
        assert len(res["per_class_accuracy"]) == 17
        assert res["confusion_matrix"].shape == (17, 17)

    def test_non_17_classes(self):
        num_classes = 5
        y_true = [0, 1, 2, 3, 4]
        y_pred = [0, 1, 2, 3, 0]

        res = evaluate_predictions(y_true, y_pred, num_classes=num_classes)
        assert len(res["per_class_accuracy"]) == 5
        assert res["confusion_matrix"].shape == (5, 5)
