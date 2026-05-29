"""Canonical default constants for the NEE classification package.

These values serve as the single source of truth and are mirrored as
defaults in the Pydantic configuration schemas.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Sites
# ---------------------------------------------------------------------------

ALL_SITES: list[str] = [
    "tumbarumba",
    "cumberland",
    "whroo",
    "wombat",
    "robson_creek_queensland",
]

# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

ALL_METRICS: list[str] = [
    "accuracy",
    "f1_macro",
    "f1_weighted",
    "precision_macro",
    "precision_weighted",
    "recall_macro",
    "recall_weighted",
]

# ---------------------------------------------------------------------------
# Column / feature names
# ---------------------------------------------------------------------------

TARGET_COL: str = "target_class"

FEATURES_TO_DROP: list[str] = [
    "NEE_VUT_REF",
    "TIMESTAMP",
]

BOM_FEATURES: list[str] = [
    "Minimum_temperature_BOM",
    "Rainfall_BOM",
    "Maximum_temperature_BOM",
]

CATEGORICAL_COLS: list[str] = [
    "year",
    "month",
    "week_of_year",
]

DATASET_FILENAME: str = "full_dataset.csv"
