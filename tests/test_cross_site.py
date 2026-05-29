"""Tests for cross-site validation logic."""

from __future__ import annotations

import pytest

from nee_classification.config.schema import DataConfig, ExperimentConfig
from nee_classification.data.cross_site import get_cross_site_splits, get_single_split


class TestCrossSiteSplits:
    """Tests for leave-one-site-out split generation."""

    def test_generates_all_splits(self):
        sites = ["a", "b", "c", "d"]
        splits = get_cross_site_splits(sites)
        assert len(splits) == 4
        for holdout, train in splits.items():
            assert holdout not in train
            assert len(train) == 3

    def test_real_sites(self):
        sites = [
            "tumbarumba", "cumberland", "whroo",
            "wombat", "robson_creek_queensland",
        ]
        splits = get_cross_site_splits(sites)
        assert len(splits) == 5
        for holdout, train in splits.items():
            assert holdout not in train
            assert set(train) | {holdout} == set(sites)


class TestSingleSplit:
    """Tests for single split extraction from config."""

    def test_valid_config(self):
        config = ExperimentConfig(
            model_type="trees",
            data=DataConfig(
                train_sites=["tumbarumba", "cumberland", "whroo", "wombat"],
                holdout_site="robson_creek_queensland",
            ),
        )
        train, holdout = get_single_split(config)
        assert holdout == "robson_creek_queensland"
        assert len(train) == 4
        assert holdout not in train

    def test_missing_holdout_raises(self):
        config = ExperimentConfig(
            model_type="trees",
            data=DataConfig(train_sites=["tumbarumba"]),
        )
        with pytest.raises(ValueError, match="holdout_site"):
            get_single_split(config)
