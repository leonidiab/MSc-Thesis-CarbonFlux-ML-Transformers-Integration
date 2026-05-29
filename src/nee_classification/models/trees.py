"""Tree-based model trainer using sklearn, XGBoost, and LightGBM."""

from __future__ import annotations

import gc
import json
import logging
import traceback
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from nee_classification.artifacts.saver import save_model_info
from nee_classification.config.schema import ExperimentConfig
from nee_classification.data.preprocessing import DataProcessor, TrainingArtifacts
from nee_classification.evaluation.metrics import evaluate_metric
from nee_classification.evaluation.reporters import (
    save_confusion_matrix_plot,
    save_excel_report,
)
from nee_classification.tuning.feature_selection import rank_features_by_importance

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model factory — replaces long if/elif chain
# ---------------------------------------------------------------------------

def _build_model(model_name: str, params: dict[str, Any]) -> Any:
    """Instantiate a tree-based classifier from name + params."""
    constructors = {
        "random_forest": RandomForestClassifier,
        "gradient_boosting": GradientBoostingClassifier,
        "extra_trees": ExtraTreesClassifier,
        "decision_tree": DecisionTreeClassifier,
    }
    # Lazy imports for optional deps
    if model_name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(**params)
    if model_name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**params)

    cls = constructors.get(model_name)
    if cls is None:
        raise ValueError(f"Unknown model: {model_name}")
    return cls(**params)


# ---------------------------------------------------------------------------
# Search space definitions (dict-of-callables)
# ---------------------------------------------------------------------------

def _suggest_params(
    trial: optuna.Trial, model_name: str, random_state: int
) -> dict[str, Any]:
    """Build Optuna search space for the given model type."""
    use_gpu = False
    try:
        import torch
        use_gpu = torch.cuda.is_available()
    except ImportError:
        pass

    spaces: dict[str, Any] = {
        "random_forest": lambda t: {
            "n_estimators": t.suggest_int("n_estimators", 100, 1000),
            "max_depth": t.suggest_int("max_depth", 5, 50),
            "min_samples_split": t.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 20),
            "max_features": t.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "bootstrap": t.suggest_categorical("bootstrap", [True, False]),
            "n_jobs": 2,
            "random_state": random_state,
        },
        "xgboost": lambda t: {
            "n_estimators": t.suggest_int("n_estimators", 100, 1000),
            "max_depth": t.suggest_int("max_depth", 3, 15),
            "learning_rate": t.suggest_float("learning_rate", 0.001, 0.5, log=True),
            "subsample": t.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": t.suggest_int("min_child_weight", 1, 20),
            "gamma": t.suggest_float("gamma", 0, 10),
            "reg_alpha": t.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": t.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": random_state,
            "n_jobs": 2,
            **({"tree_method": "hist", "device": "cuda"} if use_gpu else {}),
        },
        "lightgbm": lambda t: {
            "n_estimators": t.suggest_int("n_estimators", 100, 1000),
            "max_depth": t.suggest_int("max_depth", -1, 50),
            "learning_rate": t.suggest_float("learning_rate", 0.001, 0.5, log=True),
            "num_leaves": t.suggest_int("num_leaves", 20, 256),
            "subsample": t.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples": t.suggest_int("min_child_samples", 5, 100),
            "reg_alpha": t.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": t.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": random_state,
            "verbose": -1,
            "n_jobs": 2,
        },
        "gradient_boosting": lambda t: {
            "n_estimators": t.suggest_int("n_estimators", 100, 500),
            "max_depth": t.suggest_int("max_depth", 3, 15),
            "learning_rate": t.suggest_float("learning_rate", 0.001, 0.5, log=True),
            "min_samples_split": t.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 10),
            "subsample": t.suggest_float("subsample", 0.5, 1.0),
            "random_state": random_state,
        },
        "extra_trees": lambda t: {
            "n_estimators": t.suggest_int("n_estimators", 100, 1000),
            "max_depth": t.suggest_int("max_depth", 5, 50),
            "min_samples_split": t.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 20),
            "max_features": t.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "n_jobs": 2,
            "random_state": random_state,
        },
        "decision_tree": lambda t: {
            "max_depth": t.suggest_int("max_depth", 3, 50),
            "min_samples_split": t.suggest_int("min_samples_split", 2, 40),
            "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 20),
            "criterion": t.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
            "random_state": random_state,
        },
    }
    return spaces[model_name](trial)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class TreeModelTrainer:
    """Trainer for tree-based classifiers with Optuna hyper-parameter tuning.

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

    def run(self) -> tuple[str, float, str | None]:
        """Execute the full training pipeline.

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

        # --- Load and preprocess ----------------------------------------
        processor = DataProcessor(self.config.data, use_bom=self.use_bom)
        train_sites = self.config.data.train_sites or self.config.data.all_sites
        raw_df = processor.load_sites(train_sites)
        clean_df = processor.clean(raw_df)

        # Encode categoricals
        cat_encoders: dict[str, LabelEncoder] = {}
        for col in self.config.data.categorical_cols:
            if col in clean_df.columns:
                le = LabelEncoder()
                clean_df[col] = le.fit_transform(clean_df[col].astype(str))
                cat_encoders[col] = le

        # Split
        target_col = self.config.data.target_col
        X_all = clean_df.drop(columns=[target_col])
        y_raw = clean_df[target_col]

        le_target = LabelEncoder()
        le_target.fit(y_raw)
        y_all = le_target.transform(y_raw)

        X_train, X_val, y_train, y_val = train_test_split(
            X_all, y_all,
            test_size=self.config.val_size,
            random_state=self.random_state,
            stratify=y_all,
        )

        feature_cols = list(X_train.columns)

        # Rank features by importance for informed selection
        ranked_features = rank_features_by_importance(
            X_train, y_train, random_state=self.random_state
        )

        # --- Optuna study per model type --------------------------------
        models_to_test = self.config.model_config_extra.models_to_test
        best_results: dict[str, dict] = {}

        for model_name in models_to_test:
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=self.random_state),
            )

            def objective(trial: optuna.Trial, _mn: str = model_name) -> float:
                params = _suggest_params(trial, _mn, self.random_state)
                n_features = trial.suggest_int("n_features", 5, len(ranked_features))
                sel_features = ranked_features[:n_features]

                model = _build_model(_mn, params)
                # Internal validation split for Optuna
                X_t, X_v, y_t, y_v = train_test_split(
                    X_train[sel_features], y_train,
                    test_size=0.2, stratify=y_train,
                    random_state=self.random_state + trial.number,
                )
                model.fit(X_t, y_t)
                preds = model.predict(X_v)
                return evaluate_metric(y_v, preds, self.metric)

            study.optimize(
                objective,
                n_trials=self.config.n_trials,
                show_progress_bar=False,
            )

            try:
                best_trial = study.best_trial
                best_results[model_name] = {
                    "best_value": best_trial.value,
                    "best_params": best_trial.params,
                }
            except ValueError:
                pass

        if not best_results:
            return run_name, -1.0, "No Optuna trials completed successfully."

        # --- Select best model and retrain on full train set ------------
        best_model_name = max(
            best_results, key=lambda k: best_results[k]["best_value"]
        )
        best_info = best_results[best_model_name]
        best_params = best_info["best_params"].copy()
        n_features = best_params.pop("n_features")
        sel_features = ranked_features[:n_features]

        final_model = _build_model(best_model_name, best_params)
        final_model.fit(X_train[sel_features], y_train)

        y_val_pred = final_model.predict(X_val[sel_features])
        val_score = evaluate_metric(y_val, y_val_pred, self.metric)

        # --- Save artifacts ---------------------------------------------
        y_val_orig = le_target.inverse_transform(y_val)
        y_pred_orig = le_target.inverse_transform(y_val_pred)

        save_model_info(self.output_dir, {
            "run_name": run_name,
            "model_type": best_model_name,
            "metric_optimized": self.metric,
            "use_bom": self.use_bom,
            "best_val_score": val_score,
            "hyperparams": best_params,
            "features_selected": sel_features,
            "n_features": n_features,
            "train_sites": self.config.data.train_sites,
            "holdout_site": self.config.data.holdout_site,
            "class_mapping": dict(zip(le_target.classes_.tolist(), range(len(le_target.classes_)))),
            "random_state": self.random_state,
        })

        joblib.dump(final_model, self.output_dir / "model.joblib")
        joblib.dump(le_target, self.output_dir / "target_encoder.joblib")
        joblib.dump(cat_encoders, self.output_dir / "cat_encoders.joblib")

        # Confusion matrix
        save_confusion_matrix_plot(
            y_val_orig, y_pred_orig,
            self.output_dir / "confusion_matrix_val.png",
            title=f"Validation Confusion Matrix — {best_model_name}\n{run_name}",
        )

        # Classification report
        val_report = classification_report(
            y_val_orig, y_pred_orig, output_dict=True, zero_division=0
        )
        y_train_pred = final_model.predict(X_train[sel_features])
        y_train_orig = le_target.inverse_transform(y_train)
        y_train_pred_orig = le_target.inverse_transform(y_train_pred)
        train_report = classification_report(
            y_train_orig, y_train_pred_orig, output_dict=True, zero_division=0
        )

        save_excel_report(
            self.output_dir / "training_report.xlsx",
            features=sel_features,
            hyperparams=best_params,
            metrics_data=[
                {"split": "Train", "report": train_report},
                {"split": "Validation", "report": val_report},
            ],
        )

        logger.info(
            "Run %s complete: model=%s, %s=%.4f",
            run_name, best_model_name, self.metric, val_score,
        )
        gc.collect()
        return run_name, val_score, None
