"""Shared pytest fixtures for the nee_classification test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Minimal DataFrame mimicking the real data structure."""
    n = 40
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "TIMESTAMP": [f"2020-01-{i+1:02d}" for i in range(n)],
        "NEE_VUT_REF": rng.standard_normal(n),
        "Fpar_500m": rng.random(n),
        "Lai_500m": rng.random(n) * 5,
        "Gpp_500m": rng.random(n) * 10,
        "ET_500m": rng.random(n) * 3,
        "LE_500m": rng.random(n) * 50,
        "PET_500m": rng.random(n) * 5,
        "PLE_500m": rng.random(n) * 5,
        "LST_Day_1km": rng.random(n) * 30 + 10,
        "LST_Night_1km": rng.random(n) * 15 + 5,
        "LSTE_Day_1KM": rng.random(n) * 30 + 10,
        "LSTE_Night_1KM": rng.random(n) * 15 + 5,
        "sur_refl_b01": rng.random(n) * 1000,
        "sur_refl_b02": rng.random(n) * 1000,
        "sur_refl_b03": rng.random(n) * 1000,
        "sur_refl_b04": rng.random(n) * 1000,
        "sur_refl_b05": rng.random(n) * 1000,
        "sur_refl_b06": rng.random(n) * 1000,
        "sur_refl_b07": rng.random(n) * 1000,
        "Rainfall_BOM": rng.random(n) * 10,
        "Minimum_temperature_BOM": rng.random(n) * 10 + 5,
        "Maximum_temperature_BOM": rng.random(n) * 15 + 15,
        "target_class": ["NS"] * (n // 2) + ["S"] * (n // 2),
        "year": [2020] * n,
        "month": [1] * (n // 2) + [6] * (n // 2),
        "day_of_year": list(range(1, n + 1)),
        "week_of_year": [((i // 7) + 1) for i in range(n)],
    })


@pytest.fixture
def data_dir(tmp_path: Path, sample_dataframe: pd.DataFrame) -> Path:
    """Create a temporary data directory with sample CSVs for two sites."""
    for site in ["site_a", "site_b"]:
        site_dir = tmp_path / site
        site_dir.mkdir()
        sample_dataframe.to_csv(site_dir / "full_dataset.csv", index=False)
    return tmp_path


@pytest.fixture
def real_data_dir() -> Path:
    """Path to the real processed data directory."""
    return Path(__file__).resolve().parent.parent / "data" / "processed"
