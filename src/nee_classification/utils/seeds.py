"""Reproducibility utilities for seeding all relevant RNGs."""

from __future__ import annotations

import logging
import os
import random

import numpy as np

logger = logging.getLogger(__name__)


def seed_everything(seed: int, *, deterministic_cudnn: bool = True) -> None:
    """Seed every random-number generator used by the project.

    Seeds:
    * :mod:`random`
    * :mod:`numpy`
    * ``PYTHONHASHSEED`` environment variable
    * PyTorch (CPU **and** CUDA), if installed

    Parameters
    ----------
    seed:
        Integer seed value.
    deterministic_cudnn:
        When *True* **and** PyTorch is available, sets
        ``torch.backends.cudnn.deterministic = True`` and
        ``torch.backends.cudnn.benchmark = False`` for full
        reproducibility (may reduce performance).
    """
    logger.info("Seeding everything with seed=%d (deterministic_cudnn=%s)", seed, deterministic_cudnn)

    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002  — required for legacy code paths
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch  # type: ignore[import-untyped]

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if deterministic_cudnn and hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        logger.debug("PyTorch RNGs seeded successfully")
    except ImportError:
        logger.debug("PyTorch not installed — skipping torch seeding")
