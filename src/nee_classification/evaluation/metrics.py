"""Unified classification metrics for NEE experiments.

Provides two complementary interfaces for metric evaluation:

- :func:`evaluate_metric` — compute a metric directly from arrays.
- :func:`get_metric_from_report` — extract a metric from a
  :func:`sklearn.metrics.classification_report` dict.
"""

from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

# Supported metrics and their sklearn callables / kwargs
_METRIC_REGISTRY: dict[str, tuple] = {
    "accuracy": (accuracy_score, {}),
    "f1_macro": (f1_score, {"average": "macro"}),
    "f1_weighted": (f1_score, {"average": "weighted"}),
    "precision_macro": (precision_score, {"average": "macro", "zero_division": 0}),
    "precision_weighted": (precision_score, {"average": "weighted", "zero_division": 0}),
    "recall_macro": (recall_score, {"average": "macro", "zero_division": 0}),
    "recall_weighted": (recall_score, {"average": "weighted", "zero_division": 0}),
}

# Map from metric name prefix to classification_report key
_REPORT_METRIC_KEY = {"f1": "f1-score", "precision": "precision", "recall": "recall"}
_REPORT_AVG_KEY = {"macro": "macro avg", "weighted": "weighted avg"}


def evaluate_metric(y_true, y_pred, metric: str) -> float:
    """Compute a classification metric from true and predicted labels.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted labels.
    metric : str
        One of ``accuracy``, ``f1_macro``, ``f1_weighted``,
        ``precision_macro``, ``precision_weighted``, ``recall_macro``,
        ``recall_weighted``.

    Returns
    -------
    float
        The computed metric value.

    Raises
    ------
    ValueError
        If *metric* is not recognised.
    """
    if metric not in _METRIC_REGISTRY:
        raise ValueError(
            f"Unknown metric '{metric}'. "
            f"Supported: {sorted(_METRIC_REGISTRY)}"
        )
    fn, kwargs = _METRIC_REGISTRY[metric]
    return float(fn(y_true, y_pred, **kwargs))


def get_metric_from_report(report: dict, metric_name: str) -> float:
    """Extract a specific metric from a classification report dict.

    Parameters
    ----------
    report : dict
        Dictionary returned by ``classification_report(output_dict=True)``.
    metric_name : str
        Metric to extract (same names as :func:`evaluate_metric`).

    Returns
    -------
    float
        The extracted metric value.
    """
    if metric_name == "accuracy":
        return float(report["accuracy"])

    parts = metric_name.split("_")
    metric_type = parts[0]
    avg_type = parts[1] if len(parts) > 1 else "macro"

    avg_key = _REPORT_AVG_KEY.get(avg_type, "macro avg")
    metric_key = _REPORT_METRIC_KEY.get(metric_type, metric_type)

    return float(report[avg_key][metric_key])
