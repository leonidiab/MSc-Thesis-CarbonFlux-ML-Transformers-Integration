"""Data preprocessing pipeline for NEE classification.

The :class:`DataProcessor` orchestrates loading, cleaning, feature selection,
and label encoding.  Fitted state is captured in :class:`TrainingArtifacts`
which can be serialised to disk for reproducible test-time transforms.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from nee_classification.config.schema import DataConfig
from nee_classification.data.loader import load_sites

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training artefacts
# ---------------------------------------------------------------------------


@dataclass
class TrainingArtifacts:
    """Immutable record of fitting state needed to transform unseen data.

    Attributes
    ----------
    class_mapping:
        Mapping from original target labels to integer codes
        (e.g. ``{"NS": 0, "S": 1}``).
    feature_names:
        Ordered list of feature column names retained after dropping.
    """

    class_mapping: dict[str, int] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)

    # -- persistence -------------------------------------------------------

    def save(self, path: Path) -> None:
        """Serialise artefacts to *path* using :mod:`joblib`.

        Parameters
        ----------
        path:
            Destination file (e.g. ``artifacts/training_artifacts.joblib``).
            Parent directories are created automatically.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Training artefacts saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "TrainingArtifacts":
        """Deserialise artefacts from *path*.

        Parameters
        ----------
        path:
            File previously created by :meth:`save`.

        Returns
        -------
        TrainingArtifacts
            Restored artefacts instance.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Artefacts file not found: {path}")
        artifacts: TrainingArtifacts = joblib.load(path)
        logger.info("Training artefacts loaded from %s", path)
        return artifacts


# ---------------------------------------------------------------------------
# Data processor
# ---------------------------------------------------------------------------


class DataProcessor:
    """End-to-end data preprocessing for NEE classification experiments.

    Typical usage::

        processor = DataProcessor(config.data, use_bom=True)
        train_df = processor.load_sites(config.data.train_sites or [])
        train_df = processor.clean(train_df)
        X_train, y_train, artifacts = processor.fit_transform(train_df)

        test_df = processor.load_sites([holdout_site])
        test_df = processor.clean(test_df)
        X_test, y_test = processor.transform_test(test_df, artifacts)

    Parameters
    ----------
    config:
        Data-related configuration (paths, column names, sites).
    use_bom:
        Whether Bureau of Meteorology features should be **retained**.
        When *False* they are added to the drop list.
    """

    def __init__(self, config: DataConfig, use_bom: bool = True) -> None:
        self.config = config
        self.use_bom = use_bom
        self._drop_cols: list[str] = self._build_drop_list()
        logger.info(
            "DataProcessor initialised (use_bom=%s, dropping %d columns)",
            use_bom,
            len(self._drop_cols),
        )

    # -- private helpers ----------------------------------------------------

    def _build_drop_list(self) -> list[str]:
        """Compile the full list of columns to remove.

        Always drops ``features_to_drop_always``.  If BOM features are
        disabled the corresponding columns are appended.
        """
        drop = list(self.config.features_to_drop_always)
        if not self.use_bom:
            drop.extend(self.config.bom_features)
        return drop

    @staticmethod
    def _assert_no_missing(df: pd.DataFrame, context: str = "") -> None:
        """Raise if *df* contains any NaN values.

        Parameters
        ----------
        df:
            DataFrame to validate.
        context:
            Human-readable label included in the error message.
        """
        nan_counts = df.isna().sum()
        cols_with_nan = nan_counts[nan_counts > 0]
        if not cols_with_nan.empty:
            raise AssertionError(
                f"Unexpected NaN values in {context} data — "
                f"columns with NaN: {cols_with_nan.to_dict()}"
            )

    # -- public interface ---------------------------------------------------

    def load_sites(self, sites: list[str]) -> pd.DataFrame:
        """Load data for the given *sites*.

        Delegates to :func:`nee_classification.data.loader.load_sites`.
        """
        return load_sites(
            base_dir=self.config.base_dir,
            sites=sites,
            dataset_filename=self.config.dataset_filename,
        )

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop unwanted columns and the provenance column.

        Parameters
        ----------
        df:
            Raw dataframe as returned by :meth:`load_sites`.

        Returns
        -------
        pd.DataFrame
            Cleaned dataframe with only feature + target columns.
        """
        cols_to_drop = [c for c in self._drop_cols if c in df.columns]
        if "_source_site" in df.columns:
            cols_to_drop.append("_source_site")

        df = df.drop(columns=cols_to_drop)
        logger.info(
            "Cleaned dataframe: %d rows, %d columns (dropped %d columns)",
            len(df),
            df.shape[1],
            len(cols_to_drop),
        )
        return df

    def fit_transform(
        self,
        train_df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, TrainingArtifacts]:
        """Fit on training data and return transformed arrays + artefacts.

        1. Validates that there are **no** missing values.
        2. Builds an integer class mapping from the unique target labels
           present in *train_df*.
        3. Separates features and target, encodes the target.

        Parameters
        ----------
        train_df:
            Cleaned training dataframe (output of :meth:`clean`).

        Returns
        -------
        X_train:
            Feature matrix as a NumPy array of shape ``(n_samples, n_features)``.
        y_train:
            Integer-encoded target vector of shape ``(n_samples,)``.
        artifacts:
            Fitted :class:`TrainingArtifacts` needed for test-time transform.
        """
        self._assert_no_missing(train_df, context="training")

        target_col = self.config.target_col

        # Build class mapping from sorted unique values for determinism
        unique_classes = sorted(train_df[target_col].unique())
        class_mapping: dict[str, int] = {
            cls: idx for idx, cls in enumerate(unique_classes)
        }
        logger.info("Class mapping: %s", class_mapping)

        # Separate features and target
        feature_cols = [c for c in train_df.columns if c != target_col]

        X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
        y_train = train_df[target_col].map(class_mapping).to_numpy(dtype=np.int64)

        artifacts = TrainingArtifacts(
            class_mapping=class_mapping,
            feature_names=feature_cols,
        )

        logger.info(
            "fit_transform complete: X_train=%s, y_train=%s, %d features",
            X_train.shape,
            y_train.shape,
            len(feature_cols),
        )
        return X_train, y_train, artifacts

    def transform_test(
        self,
        test_df: pd.DataFrame,
        artifacts: TrainingArtifacts,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform test data using previously fitted artefacts.

        Parameters
        ----------
        test_df:
            Cleaned test dataframe (output of :meth:`clean`).
        artifacts:
            Artefacts produced by :meth:`fit_transform` on the training set.

        Returns
        -------
        X_test:
            Feature matrix as a NumPy array.
        y_test:
            Integer-encoded target vector.

        Raises
        ------
        AssertionError
            If NaN values are found in the test data.
        ValueError
            If the test data contains target classes not seen during training.
        """
        self._assert_no_missing(test_df, context="test")

        target_col = self.config.target_col

        # Validate target classes
        unseen = set(test_df[target_col].unique()) - set(artifacts.class_mapping)
        if unseen:
            raise ValueError(
                f"Test data contains target classes not seen during training: {unseen}"
            )

        X_test = test_df[artifacts.feature_names].to_numpy(dtype=np.float32)
        y_test = (
            test_df[target_col].map(artifacts.class_mapping).to_numpy(dtype=np.int64)
        )

        logger.info(
            "transform_test complete: X_test=%s, y_test=%s",
            X_test.shape,
            y_test.shape,
        )
        return X_test, y_test
