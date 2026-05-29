"""Experiment orchestrator — manages parallel training runs."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from nee_classification.config.schema import ExperimentConfig
from nee_classification.runner.worker import worker_task

logger = logging.getLogger(__name__)


async def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    """Run all training combinations in parallel via ProcessPoolExecutor.

    Generates the full Cartesian product of
    ``(run_idx, metric, use_bom)`` and dispatches each combination to
    :func:`worker_task` in a separate process.

    Best results are tracked **per metric** (not globally across metrics).

    Parameters
    ----------
    config : ExperimentConfig
        Complete experiment specification.

    Returns
    -------
    dict[str, Any]
        Results grouped by metric, each containing the best run info.
    """
    resources = config.resources.resolve()
    max_workers = int(resources.max_workers)

    # Build all (run_idx, metric, use_bom) combinations
    combinations: list[tuple[int, str, bool]] = []
    for metric in config.optimization_metrics:
        for use_bom in config.use_bom_options:
            for run_idx in range(1, config.n_runs + 1):
                combinations.append((run_idx, metric, use_bom))

    total = len(combinations)
    logger.info(
        "Starting experiment '%s': %d combinations, %d workers",
        config.name, total, max_workers,
    )

    # Track best per metric (NOT global cross-metric comparison)
    best_per_metric: dict[str, dict[str, Any]] = {
        m: {"score": -1.0, "run_name": "", "error": None}
        for m in config.optimization_metrics
    }

    loop = asyncio.get_event_loop()
    completed = 0
    errors: list[str] = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            loop.run_in_executor(
                executor, worker_task, run_idx, metric, use_bom, config
            ): (run_idx, metric, use_bom)
            for run_idx, metric, use_bom in combinations
        }

        async def wrap_future(f):
            res = await f
            return res, futures[f]

        for coro in asyncio.as_completed([wrap_future(f) for f in futures]):
            (run_name, score, error), (_, metric, _) = await coro
            completed += 1

            if error:
                errors.append(f"{run_name}: {error[:200]}")
                logger.warning(
                    "[%d/%d] %s FAILED: %s", completed, total, run_name, error[:100]
                )
            else:
                logger.info(
                    "[%d/%d] %s = %.4f", completed, total, run_name, score
                )
                if score > best_per_metric[metric]["score"]:
                    best_per_metric[metric]["score"] = score
                    best_per_metric[metric]["run_name"] = run_name

    # Summary
    logger.info("=" * 60)
    logger.info(
        "Experiment '%s' complete: %d/%d succeeded",
        config.name, completed - len(errors), total,
    )
    for metric, info in best_per_metric.items():
        if info["score"] > 0:
            logger.info("  Best %s: %.4f (%s)", metric, info["score"], info["run_name"])
    if errors:
        logger.warning("%d runs failed", len(errors))

    # --- Track best overall (single global winner, like original scripts) ---
    best_overall_score = -1.0
    best_overall_run_name = ""
    best_overall_combo: tuple[int, str, bool] | None = None

    for metric_name, info in best_per_metric.items():
        if info["score"] > best_overall_score:
            best_overall_score = info["score"]
            best_overall_run_name = info["run_name"]
            # Parse run_name "with_BOM/metric/run_01" to recover combo
            parts = info["run_name"].split("/")
            if len(parts) == 3:
                _use_bom = parts[0] == "with_BOM"
                _run_idx = int(parts[2].replace("run_", ""))
                best_overall_combo = (_run_idx, metric_name, _use_bom)

    # --- Cleanup: keep only best model final, like original scripts ---
    if not config.output.keep_intermediate_results and best_overall_combo is not None:
        import json
        import shutil

        holdout = config.data.holdout_site or "all_sites"
        intermediates_dir = (
            Path(config.output.results_dir) / f"holdout_{holdout}" / config.model_type
        )

        # Locate the source directory of the winning run
        run_idx, metric, use_bom = best_overall_combo
        bom_str = "with_BOM" if use_bom else "without_BOM"
        source_dir = intermediates_dir / metric / bom_str / f"run_{run_idx:02d}"

        # Destination: best_{model_type}_final inside holdout dir
        holdout_dir = Path(config.output.results_dir) / f"holdout_{holdout}"
        best_dest = holdout_dir / f"best_{config.model_type}_final"

        # Clear previous best if exists
        if best_dest.exists():
            shutil.rmtree(best_dest)
        best_dest.mkdir(parents=True, exist_ok=True)

        # Copy all artifacts from winning run
        if source_dir.exists():
            for item in source_dir.iterdir():
                dst = best_dest / item.name
                if item.is_dir():
                    shutil.copytree(item, dst)
                else:
                    shutil.copy2(item, dst)
            logger.info(
                "Best model artifacts copied from %s to %s",
                source_dir, best_dest,
            )

        # Create descriptive .txt summary (like original scripts)
        txt_path = best_dest / "best_model_summary.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"BEST {config.model_type.upper()} MODEL SUMMARY\n")
            f.write("=" * 50 + "\n")
            f.write(f"Experiment:       {config.name}\n")
            f.write(f"Holdout site:     {holdout}\n")
            f.write(f"Run:              {best_overall_run_name}\n")
            f.write(f"Validation Score: {best_overall_score:.4f}\n")
            f.write(f"BOM features:     {bom_str}\n")
            f.write(f"Metric optimized: {metric}\n")

            # Append details from info.json if available
            info_json_path = best_dest / "info.json"
            if info_json_path.exists():
                with open(info_json_path, "r", encoding="utf-8") as ij:
                    detailed = json.load(ij)
                f.write(f"Model type:       {detailed.get('model_type', 'N/A')}\n")
                f.write("\nHyperparameters:\n")
                hp = detailed.get("hyperparams", detailed.get("best_params", {}))
                f.write(json.dumps(hp, indent=4))
                feat = detailed.get("features_selected", detailed.get("feature_names", []))
                if feat:
                    f.write("\n\nSelected Features:\n")
                    f.write(", ".join(str(ft) for ft in feat))
                f.write("\n")

        logger.info("Best model summary written to %s", txt_path)

        # Also log per-metric bests to the summary
        with open(txt_path, "a", encoding="utf-8") as f:
            f.write("\n\nBEST SCORE PER METRIC\n")
            f.write("-" * 40 + "\n")
            for m, m_info in best_per_metric.items():
                if m_info["score"] > 0:
                    f.write(f"  {m}: {m_info['score']:.4f} ({m_info['run_name']})\n")

        # Delete the entire intermediates directory (like original shutil.rmtree)
        if intermediates_dir.exists():
            shutil.rmtree(intermediates_dir, ignore_errors=True)
            logger.info(
                "Intermediate results directory '%s' removed to save space.",
                intermediates_dir,
            )

    return {
        "experiment_name": config.name,
        "total_runs": total,
        "completed": completed,
        "errors": errors,
        "best_per_metric": best_per_metric,
        "best_overall": {
            "score": best_overall_score,
            "run_name": best_overall_run_name,
        },
    }
