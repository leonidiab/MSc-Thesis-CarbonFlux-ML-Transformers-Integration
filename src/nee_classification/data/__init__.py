"""Data loading and preprocessing sub-package for NEE classification.

Re-exports the primary public API so downstream code can write::

    from nee_classification.data import DataProcessor, TrainingArtifacts
"""

from __future__ import annotations

from nee_classification.data.loader import load_sites
from nee_classification.data.preprocessing import DataProcessor, TrainingArtifacts

__all__ = [
    "DataProcessor",
    "TrainingArtifacts",
    "load_sites",
]
