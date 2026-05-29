"""Tests for data preprocessing pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nee_classification.config.schema import DataConfig
from nee_classification.data.preprocessing import DataProcessor, TrainingArtifacts


class TestDataProcessor:
    """Tests for the DataProcessor class."""

    def test_clean_drops_expected_columns(self, sample_dataframe: pd.DataFrame):
        config = DataConfig(base_dir=Path("."))
        processor = DataProcessor(config, use_bom=True)
        cleaned = processor.clean(sample_dataframe)
        assert "TIMESTAMP" not in cleaned.columns
        assert "NEE_VUT_REF" not in cleaned.columns
        assert "target_class" in cleaned.columns

    def test_clean_drops_bom_when_disabled(self, sample_dataframe: pd.DataFrame):
        config = DataConfig(base_dir=Path("."))
        processor = DataProcessor(config, use_bom=False)
        cleaned = processor.clean(sample_dataframe)
        assert "Minimum_temperature_BOM" not in cleaned.columns
        assert "Rainfall_BOM" not in cleaned.columns
        assert "Maximum_temperature_BOM" not in cleaned.columns

    def test_fit_transform_returns_correct_shapes(self, sample_dataframe: pd.DataFrame):
        config = DataConfig(base_dir=Path("."))
        processor = DataProcessor(config, use_bom=True)
        cleaned = processor.clean(sample_dataframe)
        X, y, artifacts = processor.fit_transform(cleaned)

        assert X.shape[0] == len(sample_dataframe)
        assert y.shape[0] == len(sample_dataframe)
        assert X.dtype == np.float32
        assert y.dtype == np.int64

    def test_class_mapping_is_deterministic(self, sample_dataframe: pd.DataFrame):
        config = DataConfig(base_dir=Path("."))
        processor = DataProcessor(config, use_bom=True)
        cleaned = processor.clean(sample_dataframe)
        _, _, artifacts = processor.fit_transform(cleaned)

        # NS before S alphabetically
        assert artifacts.class_mapping == {"NS": 0, "S": 1}

    def test_fit_transform_rejects_nan(self, sample_dataframe: pd.DataFrame):
        config = DataConfig(base_dir=Path("."))
        processor = DataProcessor(config, use_bom=True)
        cleaned = processor.clean(sample_dataframe)
        cleaned.iloc[0, 0] = np.nan  # Inject NaN

        with pytest.raises(AssertionError, match="NaN"):
            processor.fit_transform(cleaned)

    def test_transform_test_uses_train_mapping(self, sample_dataframe: pd.DataFrame):
        config = DataConfig(base_dir=Path("."))
        processor = DataProcessor(config, use_bom=True)
        cleaned = processor.clean(sample_dataframe)
        _, _, artifacts = processor.fit_transform(cleaned)

        # Transform same data as "test"
        X_test, y_test = processor.transform_test(cleaned, artifacts)
        assert X_test.shape[0] == len(cleaned)
        assert set(np.unique(y_test)) <= set(artifacts.class_mapping.values())

    def test_transform_test_rejects_unknown_classes(self, sample_dataframe: pd.DataFrame):
        config = DataConfig(base_dir=Path("."))
        processor = DataProcessor(config, use_bom=True)
        cleaned = processor.clean(sample_dataframe)
        _, _, artifacts = processor.fit_transform(cleaned)

        # Modify test data with unknown class
        bad_test = cleaned.copy()
        bad_test.iloc[0, bad_test.columns.get_loc("target_class")] = "UNKNOWN"

        with pytest.raises(ValueError, match="not seen during training"):
            processor.transform_test(bad_test, artifacts)


class TestTrainingArtifacts:
    """Tests for TrainingArtifacts persistence."""

    def test_save_load_roundtrip(self, tmp_path: Path):
        original = TrainingArtifacts(
            class_mapping={"NS": 0, "S": 1},
            feature_names=["feat_a", "feat_b", "feat_c"],
        )
        save_path = tmp_path / "artifacts.joblib"
        original.save(save_path)

        loaded = TrainingArtifacts.load(save_path)
        assert loaded.class_mapping == original.class_mapping
        assert loaded.feature_names == original.feature_names

    def test_load_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            TrainingArtifacts.load(tmp_path / "nonexistent.joblib")


class TestRealData:
    """Tests against the actual processed datasets."""

    def test_all_sites_have_zero_nan(self, real_data_dir: Path):
        if not real_data_dir.exists():
            pytest.skip("Real data not available")

        sites = ["cumberland", "robson_creek_queensland", "tumbarumba", "whroo", "wombat"]
        for site in sites:
            csv_path = real_data_dir / site / "full_dataset.csv"
            if not csv_path.exists():
                pytest.skip(f"{csv_path} not found")
            df = pd.read_csv(csv_path)
            nan_count = df.isna().sum().sum()
            assert nan_count == 0, f"{site} has {nan_count} NaN values"

    def test_all_sites_have_same_columns(self, real_data_dir: Path):
        if not real_data_dir.exists():
            pytest.skip("Real data not available")

        sites = ["cumberland", "robson_creek_queensland", "tumbarumba", "whroo", "wombat"]
        col_sets = []
        for site in sites:
            csv_path = real_data_dir / site / "full_dataset.csv"
            if not csv_path.exists():
                pytest.skip(f"{csv_path} not found")
            df = pd.read_csv(csv_path)
            col_sets.append(set(df.columns))

        for i in range(1, len(col_sets)):
            assert col_sets[0] == col_sets[i], f"Column mismatch between site 0 and {i}"

    def test_target_column_has_expected_values(self, real_data_dir: Path):
        if not real_data_dir.exists():
            pytest.skip("Real data not available")

        csv_path = real_data_dir / "cumberland" / "full_dataset.csv"
        if not csv_path.exists():
            pytest.skip("cumberland data not found")
        df = pd.read_csv(csv_path)
        assert "target_class" in df.columns
        assert set(df["target_class"].unique()) == {"NS", "S"}

    def test_english_column_names(self, real_data_dir: Path):
        """Verify Portuguese column names were migrated to English."""
        if not real_data_dir.exists():
            pytest.skip("Real data not available")

        csv_path = real_data_dir / "cumberland" / "full_dataset.csv"
        if not csv_path.exists():
            pytest.skip("cumberland data not found")
        df = pd.read_csv(csv_path)
        assert "classe" not in df.columns, "Portuguese column 'classe' still present"
        assert "ano" not in df.columns, "Portuguese column 'ano' still present"
        assert "mes" not in df.columns, "Portuguese column 'mes' still present"
        assert "target_class" in df.columns
        assert "year" in df.columns
        assert "month" in df.columns
