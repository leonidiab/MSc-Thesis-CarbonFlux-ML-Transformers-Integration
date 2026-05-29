"""Utility sub-package for NEE classification.

Re-exports commonly used helpers for seeding and system configuration.
"""

from __future__ import annotations

from nee_classification.utils.seeds import seed_everything
from nee_classification.utils.system import (
    configure_torch_threads,
    detect_accelerator,
    resolve_max_workers,
)

__all__ = [
    "configure_torch_threads",
    "detect_accelerator",
    "resolve_max_workers",
    "seed_everything",
]
