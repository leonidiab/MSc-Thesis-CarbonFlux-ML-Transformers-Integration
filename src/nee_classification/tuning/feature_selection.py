"""Feature ranking by importance using a fast ensemble estimator."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

logger = logging.getLogger(__name__)


def rank_features_by_importance(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    feature_names: list[str] | None = None,
    *,
    random_state: int = 42,
) -> list[str]:
    """Rank features by importance using an ExtraTreesClassifier.

    Returns feature names sorted from most to least important.

    Parameters
    ----------
    X : DataFrame or ndarray
        Feature matrix.
    y : Series or ndarray
        Target labels.
    feature_names : list[str] or None
        Column names; required when *X* is an ndarray.
    random_state : int
        Seed for the estimator.

    Returns
    -------
    list[str]
        Feature names sorted by descending importance.
    """
    if isinstance(X, pd.DataFrame):
        feature_names = list(X.columns)
        X_arr = X.to_numpy()
    else:
        X_arr = X
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X_arr.shape[1])]

    clf = ExtraTreesClassifier(
        n_estimators=100, random_state=random_state, n_jobs=-1
    )
    clf.fit(X_arr, y)

    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    ranked = [feature_names[i] for i in sorted_idx]

    logger.info(
        "Feature ranking complete: top-3 = %s (importances: %s)",
        ranked[:3],
        [f"{importances[i]:.4f}" for i in sorted_idx[:3]],
    )
    return ranked
