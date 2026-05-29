"""Load and validate experiment configuration from YAML files.

Supports single-level inheritance via a special ``_inherit`` key that points
to a base YAML file (resolved relative to the child file).  An optional
*overrides* dictionary is deep-merged on top before Pydantic validation.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import yaml

from nee_classification.config.schema import ExperimentConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into a deep copy of *base*.

    For keys present in both dictionaries:

    * If both values are ``dict`` instances the merge recurses.
    * Otherwise the value from *override* wins.

    Parameters
    ----------
    base:
        Base dictionary (not mutated).
    override:
        Dictionary whose values take precedence.

    Returns
    -------
    dict[str, Any]
        Merged dictionary.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_config(
    path: Path,
    overrides: dict[str, Any] | None = None,
) -> ExperimentConfig:
    """Load an :class:`ExperimentConfig` from a YAML file.

    Parameters
    ----------
    path:
        Path to the YAML configuration file.
    overrides:
        Optional dictionary of values to deep-merge on top of the loaded
        (and possibly inherited) configuration before validation.

    Returns
    -------
    ExperimentConfig
        Fully validated experiment configuration.

    Raises
    ------
    FileNotFoundError
        If *path* (or any referenced ``_inherit`` file) does not exist.
    yaml.YAMLError
        If the file contains invalid YAML.
    pydantic.ValidationError
        If the resulting dictionary fails schema validation.
    """
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    logger.info("Loading configuration from %s", path)

    with open(path, "r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    # Handle inheritance
    if "_inherit" in raw:
        parent_path = (path.parent / raw.pop("_inherit")).resolve()
        logger.info("Inheriting base config from %s", parent_path)

        if not parent_path.is_file():
            raise FileNotFoundError(
                f"Inherited configuration file not found: {parent_path}"
            )

        with open(parent_path, "r", encoding="utf-8") as fh:
            parent_raw: dict[str, Any] = yaml.safe_load(fh) or {}

        raw = deep_merge(parent_raw, raw)

    # Apply programmatic overrides
    if overrides:
        logger.debug("Applying %d programmatic overrides", len(overrides))
        raw = deep_merge(raw, overrides)

    config = ExperimentConfig(**raw)
    logger.info("Configuration '%s' loaded and validated successfully", config.name)
    return config
