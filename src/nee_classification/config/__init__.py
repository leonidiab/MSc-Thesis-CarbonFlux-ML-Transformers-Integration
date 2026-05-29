"""Configuration sub-package for NEE classification.

Re-exports the most commonly used objects so that downstream code can write::

    from nee_classification.config import ExperimentConfig, load_config
"""

from __future__ import annotations

from nee_classification.config.loader import load_config
from nee_classification.config.schema import DataConfig, ExperimentConfig

__all__ = [
    "DataConfig",
    "ExperimentConfig",
    "load_config",
]