#!/usr/bin/env python
"""Training entry-point with full CLI support.

Usage examples::

    # Train trees with holdout wombat:
    python scripts/train.py \\
        --config configs/cross_site/holdout_wombat.yaml \\
        --model trees

    # Train FT-Transformer with custom trials:
    python scripts/train.py \\
        --config configs/cross_site/holdout_cumberland.yaml \\
        --model ft_transformer --n-trials 300 --n-runs 5
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import platform
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nee_classification import setup_logging
from nee_classification.config.loader import load_config
from nee_classification.runner.orchestrator import run_experiment

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train NEE classification models with cross-site validation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", type=Path, required=True,
        help="Path to experiment YAML configuration file.",
    )
    parser.add_argument(
        "--model", type=str, required=True,
        choices=["trees", "ft_transformer", "tabtransformer"],
        help="Model type to train.",
    )
    parser.add_argument(
        "--n-trials", type=int, default=None,
        help="Override number of Optuna trials.",
    )
    parser.add_argument(
        "--n-runs", type=int, default=None,
        help="Override number of training runs.",
    )
    parser.add_argument(
        "--max-workers", type=int, default=None,
        help="Override number of max workers for multiprocessing.",
    )
    parser.add_argument(
        "--models_to_test", type=str, nargs="+", default=None,
        help="Override models_to_test for tree models (space separated list).",
    )
    parser.add_argument(
        "--max_epochs_optuna", type=int, default=None,
        help="Override max_epochs_optuna for transformer models.",
    )
    parser.add_argument(
        "--max_epochs_final", type=int, default=None,
        help="Override max_epochs_final for transformer models.",
    )
    parser.add_argument(
        "--early_stopping_patience", type=int, default=None,
        help="Override early_stopping_patience for transformer models.",
    )
    parser.add_argument(
        "--force-cpu", action="store_true",
        help="Force CPU training even if GPU is available.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Override output directory.",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Build overrides dict from CLI args
    overrides: dict = {"model_type": args.model}
    if args.n_trials is not None:
        overrides["n_trials"] = args.n_trials
    if args.n_runs is not None:
        overrides["n_runs"] = args.n_runs
    if args.max_workers is not None:
        overrides.setdefault("resources", {})["max_workers"] = args.max_workers
    elif args.model in ("ft_transformer", "tabtransformer"):
        # Safety constraint: Neural networks consume too much RAM to run 10 parallel processes by default
        overrides.setdefault("resources", {})["max_workers"] = 1
    elif args.model == "trees":
        try:
            import torch
            if torch.cuda.is_available() and not args.force_cpu:
                # Safety constraint: XGBoost on GPU consumes too much VRAM to run 10 parallel processes
                # Cap workers at 2 to avoid CUDA Out of Memory errors
                overrides.setdefault("resources", {})["max_workers"] = 2
        except ImportError:
            pass
        
    # Model specific hyperparameter overrides
    if args.models_to_test is not None:
        overrides.setdefault("model_config_extra", {})["models_to_test"] = args.models_to_test
    if args.max_epochs_optuna is not None:
        overrides.setdefault("model_config_extra", {})["max_epochs_optuna"] = args.max_epochs_optuna
    if args.max_epochs_final is not None:
        overrides.setdefault("model_config_extra", {})["max_epochs_final"] = args.max_epochs_final
    if args.early_stopping_patience is not None:
        overrides.setdefault("model_config_extra", {})["early_stopping_patience"] = args.early_stopping_patience
    if args.force_cpu:
        overrides.setdefault("resources", {})["force_cpu"] = True
    if args.output_dir is not None:
        overrides.setdefault("output", {})["results_dir"] = str(args.output_dir)

    # Load config
    config = load_config(args.config, overrides=overrides)

    # Setup logging
    output_dir = Path(config.output.results_dir) / f"holdout_{config.data.holdout_site or 'all'}"
    setup_logging(output_dir)

    logger.info("Configuration loaded: %s", config.name)
    logger.info("Model type: %s", config.model_type)
    logger.info("Train sites: %s", config.data.train_sites)
    logger.info("Holdout site: %s", config.data.holdout_site)

    # Windows event loop policy
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Run experiment
    results = asyncio.run(run_experiment(config))

    # Print summary
    print("\n" + "=" * 60)
    print(f"Experiment: {results['experiment_name']}")
    print(f"Completed: {results['completed']}/{results['total_runs']}")
    if results["errors"]:
        print(f"Errors: {len(results['errors'])}")
    print("\nBest results per metric:")
    for metric, info in results["best_per_metric"].items():
        if info["score"] > 0:
            print(f"  {metric}: {info['score']:.4f} ({info['run_name']})")
    best = results.get("best_overall", {})
    if best.get("score", -1) > 0:
        print(f"\nBEST OVERALL: {best['score']:.4f} ({best['run_name']})")
        holdout = config.data.holdout_site or "all_sites"
        final_dir = Path(config.output.results_dir) / f"holdout_{holdout}" / f"best_{config.model_type}_final"
        print(f"Final model saved to: {final_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
