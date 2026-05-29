"""Tests for evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from nee_classification.evaluation.metrics import evaluate_metric, get_metric_from_report


class TestEvaluateMetric:
    """Tests for the evaluate_metric function."""

    def test_accuracy_perfect(self):
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 0, 1]
        assert evaluate_metric(y_true, y_pred, "accuracy") == 1.0

    def test_accuracy_half(self):
        y_true = [0, 1, 0, 1]
        y_pred = [0, 0, 0, 0]
        assert evaluate_metric(y_true, y_pred, "accuracy") == 0.5

    def test_f1_macro(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 0]
        score = evaluate_metric(y_true, y_pred, "f1_macro")
        assert 0.0 < score < 1.0

    def test_all_supported_metrics(self):
        y_true = [0, 1, 0, 1, 0, 1]
        y_pred = [0, 1, 1, 1, 0, 0]
        metrics = [
            "accuracy", "f1_macro", "f1_weighted",
            "precision_macro", "precision_weighted",
            "recall_macro", "recall_weighted",
        ]
        for m in metrics:
            score = evaluate_metric(y_true, y_pred, m)
            assert 0.0 <= score <= 1.0, f"{m} returned {score}"

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            evaluate_metric([0], [0], "nonexistent_metric")


class TestGetMetricFromReport:
    """Tests for extracting metrics from classification report dicts."""

    @pytest.fixture
    def sample_report(self) -> dict:
        return {
            "0": {"precision": 0.8, "recall": 0.9, "f1-score": 0.85, "support": 100},
            "1": {"precision": 0.7, "recall": 0.6, "f1-score": 0.65, "support": 50},
            "accuracy": 0.80,
            "macro avg": {"precision": 0.75, "recall": 0.75, "f1-score": 0.75, "support": 150},
            "weighted avg": {"precision": 0.77, "recall": 0.80, "f1-score": 0.78, "support": 150},
        }

    def test_accuracy(self, sample_report):
        assert get_metric_from_report(sample_report, "accuracy") == 0.80

    def test_f1_macro(self, sample_report):
        assert get_metric_from_report(sample_report, "f1_macro") == 0.75

    def test_precision_weighted(self, sample_report):
        assert get_metric_from_report(sample_report, "precision_weighted") == 0.77

    def test_recall_macro(self, sample_report):
        assert get_metric_from_report(sample_report, "recall_macro") == 0.75
