"""Isolated worker function for ProcessPoolExecutor."""

from __future__ import annotations

import logging
import traceback
from pathlib import Path

from nee_classification.config.schema import ExperimentConfig
from nee_classification.models.registry import get_trainer_class
from nee_classification.utils.seeds import seed_everything

logger = logging.getLogger(__name__)


def worker_task(
    run_idx: int,
    metric: str,
    use_bom: bool,
    config: ExperimentConfig,
) -> tuple[str, float, str | None]:
    """Execute a single training run in an isolated process.

    This function is the top-level callable submitted to the process pool.
    Each call seeds its own RNG, instantiates the appropriate trainer, and
    runs the full pipeline.

    Parameters
    ----------
    run_idx : int
        1-based run index.
    metric : str
        Optimisation metric name.
    use_bom : bool
        Whether BOM features are included.
    config : ExperimentConfig
        Full experiment configuration.

    Returns
    -------
    tuple[str, float, str | None]
        ``(run_name, validation_score, error_string_or_None)``
    """
    bom_str = "with_BOM" if use_bom else "without_BOM"
    holdout = config.data.holdout_site or "all_sites"

    output_dir = (
        Path(config.output.results_dir)
        / f"holdout_{holdout}"
        / config.model_type
        / metric
        / bom_str
        / f"run_{run_idx:02d}"
    )

    run_name = f"{bom_str}/{metric}/run_{run_idx:02d}"

    try:
        seed_everything(config.random_seed_base + run_idx)

        trainer_cls = get_trainer_class(config.model_type)
        trainer = trainer_cls(
            config=config,
            run_idx=run_idx,
            metric=metric,
            use_bom=use_bom,
            output_dir=output_dir,
        )

        run_name, score, error = trainer.run()
        return run_name, score, error

    except Exception:
        error = traceback.format_exc()
        logger.error("Worker %s failed:\n%s", run_name, error)
        return run_name, -1.0, error
