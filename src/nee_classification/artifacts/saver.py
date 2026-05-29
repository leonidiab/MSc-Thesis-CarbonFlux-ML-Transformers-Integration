"""Persistence utilities for model artefacts (models, scalers, metadata)."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger(__name__)


def save_model_info(output_dir: Path, info: dict[str, Any]) -> None:
    """Write experiment metadata to ``info.json``.

    Automatically adds ``python_version`` and ``data_integrity`` fields.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    info.setdefault("python_version", sys.version)
    info.setdefault("data_integrity", {"nan_count": 0})

    info_path = output_dir / "info.json"
    with open(info_path, "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=4, default=str)
    logger.info("Model info saved to %s", info_path)


def load_model_info(model_dir: Path) -> dict[str, Any]:
    """Load ``info.json`` from a model directory."""
    info_path = Path(model_dir) / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"info.json not found in {model_dir}")
    with open(info_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_sklearn_model(model: Any, path: Path) -> None:
    """Persist a scikit-learn compatible model via joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Model saved to %s", path)


def load_sklearn_model(path: Path) -> Any:
    """Load a scikit-learn compatible model from joblib file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)
