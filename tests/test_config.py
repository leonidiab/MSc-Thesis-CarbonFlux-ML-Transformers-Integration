"""Tests for configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from nee_classification.config.schema import DataConfig, ExperimentConfig, ResourceConfig
from nee_classification.config.loader import load_config


class TestDataConfig:
    """Tests for DataConfig validation."""

    def test_default_sites(self):
        config = DataConfig()
        assert len(config.all_sites) == 5
        assert "tumbarumba" in config.all_sites

    def test_valid_train_holdout(self):
        config = DataConfig(
            train_sites=["tumbarumba", "cumberland", "whroo", "wombat"],
            holdout_site="robson_creek_queensland",
        )
        assert config.holdout_site not in config.train_sites

    def test_holdout_in_train_raises(self):
        with pytest.raises(ValueError, match="must not appear in train_sites"):
            DataConfig(
                train_sites=["tumbarumba", "wombat"],
                holdout_site="wombat",
            )

    def test_unknown_train_site_raises(self):
        with pytest.raises(ValueError, match="not in all_sites"):
            DataConfig(train_sites=["nonexistent_site"])

    def test_unknown_holdout_raises(self):
        with pytest.raises(ValueError, match="not in all_sites"):
            DataConfig(holdout_site="mars_station")


class TestResourceConfig:
    """Tests for ResourceConfig resolution."""

    def test_auto_resolution(self):
        config = ResourceConfig(max_workers="auto", max_torch_threads="auto")
        resolved = config.resolve()
        assert isinstance(resolved.max_workers, int)
        assert isinstance(resolved.max_torch_threads, int)
        assert resolved.max_workers >= 1
        assert resolved.max_torch_threads >= 1

    def test_explicit_values_preserved(self):
        config = ResourceConfig(max_workers=4, max_torch_threads=8)
        resolved = config.resolve()
        assert resolved.max_workers == 4
        assert resolved.max_torch_threads == 8


class TestExperimentConfig:
    """Tests for top-level ExperimentConfig."""

    def test_minimal_valid(self):
        config = ExperimentConfig(model_type="trees")
        assert config.model_type == "trees"
        assert config.n_runs == 10
        assert config.val_size == 0.20


class TestConfigLoader:
    """Tests for YAML config loading."""

    def test_load_base_config(self):
        base_path = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"
        if not base_path.exists():
            pytest.skip("base.yaml not found")
        config = load_config(base_path, overrides={"model_type": "trees"})
        assert config.name == "base_experiment"
        assert len(config.data.all_sites) == 5

    def test_load_holdout_config(self):
        holdout_path = (
            Path(__file__).resolve().parent.parent
            / "configs" / "cross_site" / "holdout_wombat.yaml"
        )
        if not holdout_path.exists():
            pytest.skip("holdout_wombat.yaml not found")
        config = load_config(holdout_path, overrides={"model_type": "trees"})
        assert config.data.holdout_site == "wombat"
        assert "wombat" not in config.data.train_sites
