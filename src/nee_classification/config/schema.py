"""Pydantic v2 configuration schemas for NEE classification experiments.

All experiment parameters are captured in :class:`ExperimentConfig` which
composes several sub-models for data paths, compute resources, output
directories, and model-specific hyper-parameter search settings.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


class DataConfig(BaseModel):
    """Paths, column names, and site definitions for data loading."""

    base_dir: Path = Path("data/processed")
    all_sites: list[str] = Field(
        default_factory=lambda: [
            "tumbarumba",
            "cumberland",
            "whroo",
            "wombat",
            "robson_creek_queensland",
        ]
    )
    train_sites: list[str] | None = None
    holdout_site: str | None = None
    target_col: str = "target_class"
    features_to_drop_always: list[str] = Field(
        default_factory=lambda: ["NEE_VUT_REF", "TIMESTAMP"]
    )
    bom_features: list[str] = Field(
        default_factory=lambda: [
            "Minimum_temperature_BOM",
            "Rainfall_BOM",
            "Maximum_temperature_BOM",
        ]
    )
    categorical_cols: list[str] = Field(
        default_factory=lambda: ["year", "month", "week_of_year"]
    )
    dataset_filename: str = "full_dataset.csv"

    # -- cross-field validation -------------------------------------------

    @model_validator(mode="after")
    def _validate_sites(self) -> "DataConfig":
        """Ensure site lists are consistent.

        * ``train_sites`` must be a subset of ``all_sites``.
        * ``holdout_site`` must be contained in ``all_sites``.
        * ``holdout_site`` must **not** appear in ``train_sites``.
        """
        if self.train_sites is not None:
            unknown = set(self.train_sites) - set(self.all_sites)
            if unknown:
                raise ValueError(
                    f"train_sites contains sites not in all_sites: {unknown}"
                )

        if self.holdout_site is not None:
            if self.holdout_site not in self.all_sites:
                raise ValueError(
                    f"holdout_site '{self.holdout_site}' is not in all_sites"
                )

        if (
            self.train_sites is not None
            and self.holdout_site is not None
            and self.holdout_site in self.train_sites
        ):
            raise ValueError(
                f"holdout_site '{self.holdout_site}' must not appear in train_sites"
            )

        return self


class ResourceConfig(BaseModel):
    """Compute-resource budget (threads, workers, device)."""

    max_workers: int | str = "auto"
    max_torch_threads: int | str = "auto"
    force_cpu: bool = False

    def resolve(self) -> "ResourceConfig":
        """Replace ``"auto"`` sentinels with concrete integer values.

        * ``max_workers`` → ``max(1, cpu_count - 2)``
        * ``max_torch_threads`` → ``max(1, cpu_count // 2)``

        Returns
        -------
        ResourceConfig
            A **new** instance with all ``"auto"`` values resolved.
        """
        cpu_count = os.cpu_count() or 4

        resolved_max_workers: int | str = self.max_workers
        if self.max_workers == "auto":
            resolved_max_workers = max(1, cpu_count - 2)

        resolved_max_torch_threads: int | str = self.max_torch_threads
        if self.max_torch_threads == "auto":
            resolved_max_torch_threads = max(1, cpu_count // 2)

        return ResourceConfig(
            max_workers=resolved_max_workers,
            max_torch_threads=resolved_max_torch_threads,
            force_cpu=self.force_cpu,
        )


class OutputConfig(BaseModel):
    """Where results and artefacts are written."""

    results_dir: Path = Path("outputs")
    keep_intermediate_results: bool = False


class ModelSpecificConfig(BaseModel):
    """Hyper-parameter search settings shared across model families."""

    # Tree-based models
    n_trials: int = 600
    models_to_test: list[str] = Field(
        default_factory=lambda: [
            "gradient_boosting",
            "xgboost",
            "random_forest",
            "lightgbm",
            "extra_trees",
            "decision_tree",
        ]
    )

    # Transformer-based models
    max_epochs_optuna: int = 1000
    max_epochs_final: int = 5000
    early_stopping_patience: int = 50


# ---------------------------------------------------------------------------
# Top-level experiment configuration
# ---------------------------------------------------------------------------


class ExperimentConfig(BaseModel):
    """Complete specification of a single experiment run.

    Compose with :pymod:`nee_classification.config.loader` to load from YAML
    files with optional inheritance and CLI overrides.
    """

    model_config = ConfigDict(protected_namespaces=())

    name: str = "default_experiment"
    description: str = ""
    model_type: str  # "trees", "ft_transformer", "tabtransformer"
    random_seed_base: int = 42
    n_runs: int = 10
    n_trials: int = 600
    optimization_metrics: list[str] = Field(
        default_factory=lambda: [
            "accuracy",
            "f1_macro",
            "f1_weighted",
            "precision_macro",
            "precision_weighted",
            "recall_macro",
            "recall_weighted",
        ]
    )
    primary_metric: str = "f1_macro"
    use_bom_options: list[bool] = Field(default_factory=lambda: [True, False])
    val_size: float = Field(default=0.20, gt=0.0, lt=1.0)

    data: DataConfig = Field(default_factory=DataConfig)
    resources: ResourceConfig = Field(default_factory=ResourceConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    model_config_extra: ModelSpecificConfig = Field(default_factory=ModelSpecificConfig)
