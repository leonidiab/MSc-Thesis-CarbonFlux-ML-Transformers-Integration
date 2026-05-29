"""FT-Transformer trainer using pytorch-tabular."""

from __future__ import annotations

import dataclasses
import gc
import json
import logging
import os
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from nee_classification.artifacts.saver import save_model_info
from nee_classification.config.schema import ExperimentConfig
from nee_classification.data.preprocessing import DataProcessor, TrainingArtifacts
from nee_classification.evaluation.metrics import evaluate_metric
from nee_classification.evaluation.reporters import (
    save_confusion_matrix_plot,
    save_excel_report,
)

logger = logging.getLogger(__name__)


class FTTransformerTrainer:
    """Trainer for FT-Transformer models with Optuna tuning.

    Parameters
    ----------
    config : ExperimentConfig
        Full experiment configuration.
    run_idx : int
        Index of this training run (1-based).
    metric : str
        Optimisation metric name.
    use_bom : bool
        Whether BOM features are included.
    output_dir : Path
        Directory for saving all artefacts.
    """

    MODEL_CONFIG_CLASS_NAME = "FTTransformerConfig"

    def __init__(
        self,
        config: ExperimentConfig,
        run_idx: int,
        metric: str,
        use_bom: bool,
        output_dir: Path,
    ) -> None:
        self.config = config
        self.run_idx = run_idx
        self.metric = metric
        self.use_bom = use_bom
        self.output_dir = Path(output_dir)
        self.random_state = config.random_seed_base + run_idx

    # ------------------------------------------------------------------
    # Model config creation (override in TabTransformerTrainer)
    # ------------------------------------------------------------------

    def _create_model_config(
        self, trial: optuna.Trial, n_features: int
    ) -> Any:
        """Create a pytorch-tabular model config from Optuna suggestions."""
        from pytorch_tabular.config import OptimizerConfig, TrainerConfig
        from pytorch_tabular.models import FTTransformerConfig

        num_attn_blocks = trial.suggest_int("num_attn_blocks", 2, 6)
        num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        attn_dropout = trial.suggest_float("attn_dropout", 0.0, 0.3)
        ff_dropout = trial.suggest_float("ff_dropout", 0.0, 0.5)
        embed_dim = trial.suggest_categorical("embed_dim", [32, 64, 128])
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

        max_epochs = self.config.model_config_extra.max_epochs_optuna
        patience = self.config.model_config_extra.early_stopping_patience

        return FTTransformerConfig(
            task="classification",
            input_embed_dim=embed_dim,
            num_attn_blocks=num_attn_blocks,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            learning_rate=lr,
            metrics=["accuracy"],
            metrics_prob_input=[False],
        ), batch_size, max_epochs, patience

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def run(self) -> tuple[str, float, str | None]:
        """Execute the full FT-Transformer training pipeline.

        Returns
        -------
        tuple[str, float, str | None]
            ``(run_name, best_val_score, error_string_or_None)``
        """
        bom_str = "with_BOM" if self.use_bom else "without_BOM"
        run_name = f"{bom_str}/{self.metric}/run_{self.run_idx:02d}"

        try:
            return self._run_inner(run_name, bom_str)
        except Exception:
            error = traceback.format_exc()
            logger.error("Run %s failed:\n%s", run_name, error)
            return run_name, -1.0, error

    def _run_inner(
        self, run_name: str, bom_str: str
    ) -> tuple[str, float, str | None]:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Lazy imports (avoids torch import if not needed)
        import torch
        from pytorch_tabular import TabularModel
        from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig

        accelerator = "cpu"
        if torch.cuda.is_available() and not self.config.resources.force_cpu:
            accelerator = "auto"

        # --- Load and preprocess -----------------------------------------
        processor = DataProcessor(self.config.data, use_bom=self.use_bom)
        train_sites = self.config.data.train_sites or self.config.data.all_sites
        raw_df = processor.load_sites(train_sites)
        clean_df = processor.clean(raw_df)

        target_col = self.config.data.target_col
        X_all = clean_df.drop(columns=[target_col])
        y_all = clean_df[target_col]

        # Class mapping
        train_classes = sorted(y_all.unique())
        class_mapping = {cls: idx for idx, cls in enumerate(train_classes)}
        clean_df[target_col] = clean_df[target_col].map(class_mapping)

        # Split
        df_train, df_val = train_test_split(
            clean_df,
            test_size=self.config.val_size,
            random_state=self.random_state,
            stratify=clean_df[target_col],
        )

        # Normalize numerical features
        cat_cols = [c for c in self.config.data.categorical_cols if c in df_train.columns]
        num_cols = [c for c in df_train.columns if c != target_col and c not in cat_cols]

        scaler = StandardScaler()
        df_train[num_cols] = scaler.fit_transform(df_train[num_cols])
        df_val[num_cols] = scaler.transform(df_val[num_cols])

        # Determine batch_size safety
        n_train = len(df_train)

        # --- Optuna study -----------------------------------------------
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )

        def objective(trial: optuna.Trial) -> float:
            model = None
            try:
                # Feature selection: select subset of numerical features
                selected_num = []
                for feat in num_cols:
                    if trial.suggest_int(f"use_{feat}", 0, 1) == 1:
                        selected_num.append(feat)

                if not selected_num:
                    selected_num = num_cols[:5]

                all_features = selected_num + cat_cols
                train_subset = df_train[all_features + [target_col]].copy()
                val_subset = df_val[all_features + [target_col]].copy()

                model_config, batch_size, max_epochs, patience = (
                    self._create_model_config(trial, len(all_features))
                )

                # Adjust batch_size for small datasets
                safe_batch = min(batch_size, n_train - 1)
                if safe_batch < 2:
                    safe_batch = 2

                data_config = DataConfig(
                    target=[target_col],
                    continuous_cols=selected_num,
                    categorical_cols=cat_cols,
                )
                trainer_config = TrainerConfig(
                    batch_size=safe_batch,
                    max_epochs=max_epochs,
                    early_stopping="valid_loss",
                    early_stopping_patience=patience,
                    accelerator=accelerator,
                    checkpoints=None,
                    progress_bar="none",
                )
                optimizer_config = OptimizerConfig()

                model = TabularModel(
                    data_config=data_config,
                    model_config=model_config,
                    optimizer_config=optimizer_config,
                    trainer_config=trainer_config,
                )
                model.fit(train=train_subset, validation=val_subset)

                result = model.evaluate(val_subset, verbose=False)
                preds = model.predict(val_subset)
                pred_col = f"{target_col}_prediction"
                y_pred = preds[pred_col].values
                y_true = val_subset[target_col].values

                score = evaluate_metric(y_true, y_pred, self.metric)
                return score

            except Exception as e:
                logger.warning("Trial %d failed: %s", trial.number, e)
                err_path = self.output_dir / "error_log.txt"
                with open(err_path, "a", encoding="utf-8") as fh:
                    fh.write(f"Trial {trial.number}: {traceback.format_exc()}\n")
                raise optuna.exceptions.TrialPruned()
            finally:
                if model is not None:
                    del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            show_progress_bar=False,
        )

        try:
            best_trial = study.best_trial
        except ValueError:
            return run_name, -1.0, "No trials completed."

        best_params = best_trial.params
        val_score = best_trial.value

        # Extract selected features from best trial params
        best_num_features = [
            f for f in num_cols if best_params.get(f"use_{f}", 0) == 1
        ]
        if not best_num_features:
            best_num_features = num_cols[:5]
        best_features = best_num_features + cat_cols

        # Derive hyperparams (non-feature params)
        hyperparams = {
            k: v for k, v in best_params.items() if not k.startswith("use_")
        }

        # --- Save artifacts -----------------------------------------------
        save_model_info(self.output_dir, {
            "run_name": run_name,
            "model_type": self.MODEL_CONFIG_CLASS_NAME,
            "metric_optimized": self.metric,
            "use_bom": self.use_bom,
            "best_val_score": val_score,
            "best_params": best_params,
            "hyperparams": hyperparams,
            "features_selected": best_num_features,
            "class_mapping": class_mapping,
            "n_trials": self.config.n_trials,
            "train_sites": self.config.data.train_sites,
            "holdout_site": self.config.data.holdout_site,
            "numerical_features": num_cols,
            "categorical_features": cat_cols,
            "random_state": self.random_state,
        })

        # Save scaler
        import joblib
        joblib.dump(scaler, self.output_dir / "scaler.joblib")

        # Save training artifacts
        artifacts = TrainingArtifacts(
            class_mapping=class_mapping,
            feature_names=num_cols + cat_cols,
        )
        artifacts.save(self.output_dir / "training_artifacts.joblib")

        # --- Generate validation predictions for reports -------------------
        # Re-run prediction on val with the best trial params to get y_true/y_pred
        # (We use the score already computed by Optuna, but need preds for confusion matrix)
        inverse_mapping = {v: k for k, v in class_mapping.items()}
        val_subset = df_val[best_features + [target_col]].copy()
        train_subset = df_train[best_features + [target_col]].copy()

        try:
            model_config, batch_size, max_epochs, patience = (
                self._create_model_config(best_trial, len(best_features))
            )
            safe_batch = min(batch_size, n_train - 1)
            if safe_batch < 2:
                safe_batch = 2

            data_config_final = DataConfig(
                target=[target_col],
                continuous_cols=best_num_features,
                categorical_cols=cat_cols,
            )
            trainer_config_final = TrainerConfig(
                batch_size=safe_batch,
                max_epochs=self.config.model_config_extra.max_epochs_final,
                early_stopping="valid_loss",
                early_stopping_patience=self.config.model_config_extra.early_stopping_patience,
                accelerator=accelerator,
                checkpoints=None,
                progress_bar="none",
            )

            final_model = TabularModel(
                data_config=data_config_final,
                model_config=model_config,
                optimizer_config=OptimizerConfig(),
                trainer_config=trainer_config_final,
            )
            final_model.fit(train=train_subset, validation=val_subset)

            # Save the pytorch-tabular model
            final_model.save_model(self.output_dir / "model_final")

            # Get early stopping epoch
            early_stop_epoch: int | str = "N/A"
            if hasattr(final_model, "trainer") and final_model.trainer:
                for cb in final_model.trainer.callbacks:
                    if cb.__class__.__name__ == "EarlyStopping" and getattr(cb, "stopped_epoch", 0) > 0:
                        early_stop_epoch = cb.stopped_epoch + 1
                        break

            val_preds = final_model.predict(val_subset)
            pred_col = f"{target_col}_prediction"
            y_pred = val_preds[pred_col].values
            y_true = val_subset[target_col].values

            # Map back to original class labels for reports
            y_true_labels = [inverse_mapping.get(v, v) for v in y_true]
            y_pred_labels = [inverse_mapping.get(v, v) for v in y_pred]

            # Confusion matrix
            save_confusion_matrix_plot(
                y_true_labels, y_pred_labels,
                self.output_dir / "confusion_matrix_val.png",
                title=f"Validation Confusion Matrix — {self.MODEL_CONFIG_CLASS_NAME}\n{run_name}",
            )

            # Excel report
            val_report = classification_report(
                y_true_labels, y_pred_labels, output_dict=True, zero_division=0,
            )
            train_preds = final_model.predict(train_subset)
            y_train_pred = train_preds[pred_col].values
            y_train_true = train_subset[target_col].values
            y_train_true_labels = [inverse_mapping.get(v, v) for v in y_train_true]
            y_train_pred_labels = [inverse_mapping.get(v, v) for v in y_train_pred]

            train_report = classification_report(
                y_train_true_labels, y_train_pred_labels, output_dict=True, zero_division=0,
            )

            save_excel_report(
                self.output_dir / "training_report.xlsx",
                features=best_num_features,
                hyperparams={**hyperparams, "early_stopping_epoch": early_stop_epoch},
                metrics_data=[
                    {"split": "Train", "report": train_report},
                    {"split": "Validation", "report": val_report},
                ],
            )

            # Update val_score with the final model's score
            val_score = evaluate_metric(y_true, y_pred, self.metric)

        except Exception as e:
            logger.warning("Final model report generation failed: %s", e)
        finally:
            if "final_model" in dir():
                del final_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info(
            "Run %s complete: %s=%.4f (%s)",
            run_name, self.metric, val_score, self.MODEL_CONFIG_CLASS_NAME,
        )

        gc.collect()
        return run_name, val_score, None
