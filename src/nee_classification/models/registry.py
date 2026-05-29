"""Model registry — lazy-loading trainer classes by name."""

from __future__ import annotations

import importlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, str] = {
    "trees": "nee_classification.models.trees.TreeModelTrainer",
    "ft_transformer": "nee_classification.models.ft_transformer.FTTransformerTrainer",
    "tabtransformer": "nee_classification.models.tabtransformer.TabTransformerTrainer",
}


def get_trainer_class(model_type: str) -> Any:
    """Return the trainer class for *model_type* (lazy import).

    Parameters
    ----------
    model_type : str
        One of ``"trees"``, ``"ft_transformer"``, ``"tabtransformer"``.

    Returns
    -------
    type
        The trainer class.

    Raises
    ------
    ValueError
        If *model_type* is unknown.
    """
    qualname = _REGISTRY.get(model_type)
    if qualname is None:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Available: {sorted(_REGISTRY)}"
        )

    module_path, cls_name = qualname.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, cls_name)
    logger.debug("Loaded trainer class: %s", qualname)
    return cls
