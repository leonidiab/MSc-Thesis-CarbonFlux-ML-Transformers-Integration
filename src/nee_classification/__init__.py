"""NEE Classification — scientific ML package for Net Ecosystem Exchange classification.

Provides utilities for data loading, preprocessing, configuration management,
and model training for classifying NEE flux tower observations.
"""

from __future__ import annotations

import logging
from pathlib import Path

__version__ = "1.0.0"

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging(log_dir: Path, level: int = logging.INFO) -> None:
    """Configure root logger with file and console handlers.

    Creates a log file named ``nee_classification.log`` inside *log_dir* and
    attaches both a :class:`~logging.FileHandler` and a
    :class:`~logging.StreamHandler` to the root logger.  Noisy third-party
    loggers (``pytorch_lightning``, ``optuna``) are silenced to
    :data:`~logging.WARNING`.

    Parameters
    ----------
    log_dir:
        Directory where the log file will be written.  Created automatically
        if it does not already exist.
    level:
        Logging level for the root logger.  Defaults to :data:`logging.INFO`.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Avoid adding duplicate handlers on repeated calls
    if root_logger.handlers:
        root_logger.handlers.clear()

    formatter = logging.Formatter(_LOG_FORMAT)

    # File handler
    file_handler = logging.FileHandler(log_dir / "nee_classification.log")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Silence noisy third-party loggers
    for noisy_logger_name in ("pytorch_lightning", "optuna"):
        logging.getLogger(noisy_logger_name).setLevel(logging.WARNING)

    root_logger.debug("Logging initialised — log file: %s", log_dir / "nee_classification.log")
