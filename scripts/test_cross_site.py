#!/usr/bin/env python
"""Cross-site test entry-point.

Usage::

    python scripts/test_cross_site.py \\
        --model-dir outputs/holdout_wombat/trees/f1_macro/with_BOM/run_01 \\
        --test-site wombat

    python scripts/test_cross_site.py \\
        --model-dir outputs/holdout_wombat/ft_transformer/accuracy/with_BOM/run_01 \\
        --test-site wombat --data-dir data/processed
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nee_classification import setup_logging
from nee_classification.artifacts.saver import load_model_info, load_sklearn_model
from nee_classification.data.loader import load_sites
from nee_classification.data.preprocessing import TrainingArtifacts
from nee_classification.evaluation.reporters import (
    save_classification_report,
    save_confusion_matrix_plot,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Test a trained model on a holdout site.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model-dir", type=Path, required=True,
        help="Directory containing trained model and artifacts.",
    )
    parser.add_argument(
        "--test-site", type=str, required=True,
        help="Site name to use as test set.",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/processed"),
        help="Base directory containing site data folders.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Where to save test results (default: model-dir/test_results/).",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for cross-site testing."""
    args = parse_args()
    model_dir = args.model_dir
    test_site = args.test_site
    output_dir = args.output_dir or (model_dir / "test_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(output_dir)

    # --- Load model info -------------------------------------------------
    info = load_model_info(model_dir)
    logger.info("Model info loaded from %s", model_dir)
    logger.info("Model type: %s", info.get("model_type", "unknown"))
    logger.info("Metric optimized: %s", info.get("metric_optimized", "unknown"))

    # --- Load test data --------------------------------------------------
    df_test = load_sites(args.data_dir, [test_site])
    logger.info("Test data loaded: %s (%d rows)", test_site, len(df_test))

    # --- Check for NaN ---------------------------------------------------
    na_count = df_test.isna().sum().sum()
    if na_count > 0:
        na_cols = df_test.columns[df_test.isna().any()].tolist()
        raise ValueError(
            f"Unexpected NaN in test data: {na_count} NaN in columns {na_cols}"
        )

    # --- Prepare features ------------------------------------------------
    features = info.get("features_selected") or info.get("feature_names", [])
    class_mapping = info.get("class_mapping", {})

    # Drop columns not needed
    target_col = "target_class"
    if "_source_site" in df_test.columns:
        df_test = df_test.drop(columns=["_source_site"])

    # Handle categorical encoders for tree models
    cat_encoders_path = model_dir / "cat_encoders.joblib"
    if cat_encoders_path.exists():
        cat_encoders = joblib.load(cat_encoders_path)
        for col, le in cat_encoders.items():
            if col in df_test.columns:
                df_test[col] = le.transform(df_test[col].astype(str))

    # Handle scaler for transformer models
    scaler_path = model_dir / "scaler.joblib"
    num_features = info.get("numerical_features", [])
    if scaler_path.exists() and num_features:
        scaler = joblib.load(scaler_path)
        cols_to_scale = [c for c in num_features if c in df_test.columns]
        if cols_to_scale:
            df_test[cols_to_scale] = scaler.transform(df_test[cols_to_scale])

    # --- Get true labels --------------------------------------------------
    y_test_raw = df_test[target_col].values

    # Encode target with training class mapping
    if class_mapping:
        y_test = np.array([class_mapping.get(str(v), -1) for v in y_test_raw])
    else:
        le_path = model_dir / "target_encoder.joblib"
        if le_path.exists():
            le_target = joblib.load(le_path)
            y_test = le_target.transform(y_test_raw)
        else:
            raise FileNotFoundError("No class_mapping or target_encoder found.")

    # --- Load and predict ------------------------------------------------
    model_path = model_dir / "model.joblib"
    model_final_path = model_dir / "model_final"
    if model_path.exists():
        # sklearn / tree model
        model = load_sklearn_model(model_path)
        missing_feats = [f for f in features if f not in df_test.columns]
        if missing_feats:
            raise ValueError(f"Features missing in test data: {missing_feats}")
        X_test = df_test[features].values
        y_pred = model.predict(X_test)
    elif model_final_path.exists():
        # pytorch-tabular model
        from pytorch_tabular import TabularModel
        
        logger.info("Loading PyTorch Tabular model from %s", model_final_path)
        model = TabularModel.load_model(str(model_final_path))
        
        missing_feats = [f for f in features if f not in df_test.columns]
        if missing_feats:
            raise ValueError(f"Features missing in test data: {missing_feats}")
        
        # Prepare test DataFrame with same targets encoded
        df_test_proc = df_test.copy()
        df_test_proc[target_col] = y_test
        
        # Predict using TabularModel
        preds = model.predict(df_test_proc)
        pred_col = f"{target_col}_prediction"
        if pred_col not in preds.columns:
            pred_cols = [c for c in preds.columns if c.endswith("_prediction")]
            if pred_cols:
                pred_col = pred_cols[0]
            else:
                raise ValueError(f"Could not find prediction column in model predictions. Columns: {preds.columns}")
        y_pred = preds[pred_col].values
    else:
        raise FileNotFoundError(
            f"No loadable model found in {model_dir}. "
            "Expected model.joblib or model_final/ directory."
        )

    # --- Evaluate --------------------------------------------------------
    # Inverse map for readable labels
    inv_mapping = {v: k for k, v in class_mapping.items()} if class_mapping else {}
    if inv_mapping:
        y_test_labels = [inv_mapping.get(v, str(v)) for v in y_test]
        y_pred_labels = [inv_mapping.get(v, str(v)) for v in y_pred]
    else:
        le_target = joblib.load(model_dir / "target_encoder.joblib")
        y_test_labels = le_target.inverse_transform(y_test)
        y_pred_labels = le_target.inverse_transform(y_pred)

    report_dict = classification_report(
        y_test_labels, y_pred_labels, output_dict=True, zero_division=0
    )
    report_str = classification_report(
        y_test_labels, y_pred_labels, zero_division=0
    )

    # --- Print and save --------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"CROSS-SITE TEST REPORT — {test_site.upper()}")
    print(f"Model: {info.get('model_type', 'unknown')}")
    print(f"Trained with: {info.get('train_sites', 'unknown')}")
    print(f"{'=' * 60}\n")
    print(report_str)

    # Save JSON report
    save_classification_report(
        report_dict,
        output_dir / f"classification_report_{test_site}.json",
    )

    # Save confusion matrix (name derived from test_site, NOT hardcoded)
    save_confusion_matrix_plot(
        y_test_labels, y_pred_labels,
        output_dir / f"confusion_matrix_{test_site}.png",
        title=f"Confusion Matrix — Test Site: {test_site.replace('_', ' ').title()}",
    )

    logger.info("Test results saved to %s", output_dir)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
