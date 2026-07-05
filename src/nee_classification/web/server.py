import os
import sys
import json
import logging
from pathlib import Path
import base64
import io

import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, send_from_directory
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server threads
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to python path for local imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nee_classification.artifacts.saver import load_model_info, load_sklearn_model
from nee_classification.data.preprocessing import TrainingArtifacts

app = Flask(__name__, static_folder="static", template_folder="templates")

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Security Configuration
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB limit
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH


def safe_resolve_path(base_dir: Path, user_path_str: str) -> Path:
    """Resolve user-supplied path relative to base_dir, protecting against path traversal."""
    resolved_base = base_dir.resolve()
    target_path = (resolved_base / user_path_str).resolve()
    
    # Ensure target is within the base directory
    if not str(target_path).startswith(str(resolved_base)):
        raise PermissionError("Access denied: Path traversal detected.")
    return target_path


@app.after_request
def apply_security_headers(response):
    """Apply crucial security headers to all responses."""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Content-Security-Policy'] = "default-src 'self'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src https://fonts.gstatic.com; img-src 'self' data:; script-src 'self' 'unsafe-inline';"
    return response


@app.route("/")
def index():
    """Serve the single-page application dashboard."""
    return send_from_directory(app.template_folder, "index.html")


@app.route("/api/models", methods=["GET"])
def list_models():
    """Dynamically scan outputs/ folder and return metadata for all trained models."""
    try:
        models = []
        if not OUTPUTS_DIR.exists():
            return jsonify([])

        # Scan directory recursively
        for path in OUTPUTS_DIR.rglob("info.json"):
            model_dir = path.parent
            # Check if there is a loadable model
            has_joblib = (model_dir / "model.joblib").is_file()
            has_pytorch_tabular = (model_dir / "model_final").is_dir()
            
            if has_joblib or has_pytorch_tabular:
                try:
                    info = load_model_info(model_dir)
                    # Extract relative path to allow client reference without exposing full system paths
                    rel_dir = os.path.relpath(model_dir, OUTPUTS_DIR)
                    
                    models.append({
                        "id": rel_dir.replace("\\", "/"),
                        "model_type": info.get("model_type", "unknown"),
                        "metric_optimized": info.get("metric_optimized", "unknown"),
                        "best_val_score": info.get("best_val_score", 0.0),
                        "use_bom": info.get("use_bom", True),
                        "features": info.get("features_selected") or info.get("feature_names", []),
                        "train_sites": info.get("train_sites", []),
                        "holdout_site": info.get("holdout_site", "unknown"),
                        "class_mapping": info.get("class_mapping", {"NS": 0, "S": 1})
                    })
                except Exception as e:
                    logger.error(f"Error loading model info at {model_dir}: {e}")
                    
        return jsonify(models)
    except Exception as e:
        logger.exception("Failed to scan models directory")
        return jsonify({"error": "Failed to load models list."}), 500


def get_confusion_matrix_base64(y_true, y_pred, labels):
    """Generate a confusion matrix heatmap and return it as a base64 encoded PNG."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(5, 4))
    sns.set_theme(style="dark")
    
    # Custom color palette matching the UI's dark aesthetics
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar=False,
                annot_kws={"size": 14, "weight": "bold"},
                facecolor='#1e293b')
    
    plt.title('Confusion Matrix', color='#f8fafc', fontsize=14, pad=10)
    plt.ylabel('True Class', color='#94a3b8')
    plt.xlabel('Predicted Class', color='#94a3b8')
    plt.tick_params(colors='#94a3b8')
    
    # Set background colors for glassmorphism integration
    fig = plt.gcf()
    fig.patch.set_facecolor('#0f172a')
    ax = plt.gca()
    ax.set_facecolor('#0f172a')
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_b64


def prepare_data_inference(df: pd.DataFrame, model_dir: Path, info: dict) -> pd.DataFrame:
    """Preprocess data following the pipeline, with security and structural guards."""
    df_proc = df.copy()
    
    # Handle categorical encoders for tree models
    cat_encoders_path = model_dir / "cat_encoders.joblib"
    if cat_encoders_path.exists():
        cat_encoders = joblib.load(cat_encoders_path)
        for col, le in cat_encoders.items():
            if col not in df_proc.columns:
                df_proc[col] = le.classes_[0]
            
            # Map unseen category values gracefully to the first known class
            s_data = df_proc[col].astype(str)
            known_classes = set(le.classes_)
            s_data_mapped = s_data.apply(lambda x: x if x in known_classes else le.classes_[0])
            df_proc[col] = le.transform(s_data_mapped)

    # Handle scaler for transformer models
    scaler_path = model_dir / "scaler.joblib"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        # Determine fitted feature names and their exact order
        if hasattr(scaler, "feature_names_in_"):
            scale_features = list(scaler.feature_names_in_)
        else:
            scale_features = info.get("numerical_features", [])
            
        if scale_features:
            # Ensure all features expected by the scaler are present
            for col in scale_features:
                if col not in df_proc.columns:
                    df_proc[col] = 0.0
            
            # Run transformation on the exact columns in the exact fitted order
            df_proc[scale_features] = scaler.transform(df_proc[scale_features])
            
    return df_proc


@app.route("/api/predict", methods=["POST"])
def predict():
    """Run model inference on single data instances or batch CSVs."""
    try:
        model_id = request.form.get("model_id")
        input_type = request.form.get("input_type", "single")
        
        if not model_id:
            return jsonify({"error": "Missing model_id parameter."}), 400
            
        # Resolve and validate model directory path strictly
        try:
            model_dir = safe_resolve_path(OUTPUTS_DIR, model_id)
        except PermissionError as pe:
            return jsonify({"error": str(pe)}), 403
            
        if not model_dir.is_dir():
            return jsonify({"error": "Selected model directory does not exist."}), 404
            
        info = load_model_info(model_dir)
        features = info.get("features_selected") or info.get("feature_names", [])
        class_mapping = info.get("class_mapping", {"NS": 0, "S": 1})
        inv_mapping = {v: k for k, v in class_mapping.items()}
        
        # Load Model
        model_path = model_dir / "model.joblib"
        model_final_path = model_dir / "model_final"
        is_tabular = False
        
        if model_path.exists():
            model = load_sklearn_model(model_path)
        elif model_final_path.exists():
            from pytorch_tabular import TabularModel
            model = TabularModel.load_model(str(model_final_path))
            is_tabular = True
        else:
            return jsonify({"error": "No trained model artifacts found in directory."}), 404
            
        # Compile input dataset
        if input_type == "single":
            single_data = request.form.get("data")
            if not single_data:
                return jsonify({"error": "Missing input data."}), 400
            inputs = json.loads(single_data)
            
            # Form single-row DataFrame
            df_input = pd.DataFrame([inputs])
            
            # Fill missing columns with reasonable defaults (0) if any
            all_required_cols = list(features)
            if is_tabular:
                all_required_cols = list(set(
                    all_required_cols + 
                    list(model.config.categorical_cols) + 
                    list(model.config.continuous_cols)
                ))
            
            for f in all_required_cols:
                if f not in df_input.columns:
                    if is_tabular and f in model.config.categorical_cols:
                        df_input[f] = 0
                    else:
                        df_input[f] = 0.0
            
            df_processed = prepare_data_inference(df_input, model_dir, info)
            
            # Predict
            if not is_tabular:
                X = df_processed[features].values
                pred_code = int(model.predict(X)[0])
                prob = None
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X)[0]
                    prob = {inv_mapping.get(i, f"Class {i}"): float(p) for i, p in enumerate(probs)}
            else:
                # Add target dummy for TabularModel input compatibility
                df_processed["target_class"] = 0
                preds = model.predict(df_processed)
                
                pred_col = "target_class_prediction"
                if pred_col not in preds.columns:
                    pred_cols = [c for c in preds.columns if c.endswith("_prediction")]
                    if pred_cols:
                        pred_col = pred_cols[0]
                pred_code = int(preds[pred_col].values[0])
                
                # Fetch probabilities if available
                prob = {}
                target_col = "target_class"
                for cls_name, cls_idx in class_mapping.items():
                    patterns = [
                        f"{cls_idx}_probability",
                        f"{target_col}_{cls_idx}_probability",
                        f"target_class_{cls_idx}_probability"
                    ]
                    for pat in patterns:
                        if pat in preds.columns:
                            prob[cls_name] = float(preds[pat].values[0])
                            break
            
            prediction_label = inv_mapping.get(pred_code, str(pred_code))
            return jsonify({
                "prediction": prediction_label,
                "probabilities": prob
            })
            
        elif input_type == "file":
            if 'file' not in request.files:
                return jsonify({"error": "No file uploaded."}), 400
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "Selected file is empty."}), 400
                
            # Read CSV safely with pandas
            try:
                df_raw = pd.read_csv(file)
            except Exception as csv_err:
                return jsonify({"error": f"Failed to parse CSV: {str(csv_err)}"}), 400
                
            # Check features presence
            missing_features = [f for f in features if f not in df_raw.columns]
            if missing_features:
                return jsonify({
                    "error": f"CSV is missing columns required by the model: {missing_features}"
                }), 400
                
            # Drop NaNs or warn
            nan_count = df_raw[features].isna().sum().sum()
            if nan_count > 0:
                # Standardize NaN handling: fill NaNs with column median to prevent crashes
                for f in features:
                    if df_raw[f].isna().any():
                        median_val = df_raw[f].median()
                        df_raw[f] = df_raw[f].fillna(median_val)
                        
            # For tabular models, guarantee that all expected categorical/continuous config columns are present
            if is_tabular:
                all_required_cols = list(set(
                    list(model.config.categorical_cols) + 
                    list(model.config.continuous_cols)
                ))
                for f in all_required_cols:
                    if f not in df_raw.columns:
                        if f in model.config.categorical_cols:
                            df_raw[f] = 0
                        else:
                            df_raw[f] = 0.0

            df_processed = prepare_data_inference(df_raw, model_dir, info)
            
            # Predict batch
            if not is_tabular:
                X = df_processed[features].values
                pred_codes = model.predict(X)
                probs = model.predict_proba(X) if hasattr(model, "predict_proba") else None
            else:
                df_processed["target_class"] = 0
                preds = model.predict(df_processed)
                
                pred_col = "target_class_prediction"
                if pred_col not in preds.columns:
                    pred_cols = [c for c in preds.columns if c.endswith("_prediction")]
                    if pred_cols:
                        pred_col = pred_cols[0]
                pred_codes = preds[pred_col].values
                
                probs = []
                target_col = "target_class"
                for idx in range(len(preds)):
                    row_probs = []
                    for cls_idx in sorted(class_mapping.values()):
                        found_val = None
                        patterns = [
                            f"{cls_idx}_probability",
                            f"{target_col}_{cls_idx}_probability",
                            f"target_class_{cls_idx}_probability"
                        ]
                        for pat in patterns:
                            if pat in preds.columns:
                                found_val = preds.iloc[idx][pat]
                                break
                        if found_val is not None:
                            row_probs.append(found_val)
                    if row_probs:
                        probs.append(row_probs)
                probs = np.array(probs) if probs else None
                
            # Map predictions to labels
            pred_labels = [inv_mapping.get(int(code), str(code)) for code in pred_codes]
            
            # Append predictions back to df_raw for export/download
            df_output = df_raw.copy()
            df_output["predicted_class"] = pred_labels
            if probs is not None:
                for cls_name, cls_idx in class_mapping.items():
                    if cls_idx < probs.shape[1]:
                        df_output[f"probability_{cls_name}"] = probs[:, cls_idx]
            
            # Convert output to CSV string for download
            csv_buffer = io.StringIO()
            df_output.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            # Optional metrics if target column is present in uploaded CSV
            has_ground_truth = "target_class" in df_raw.columns
            metrics = None
            confusion_matrix_img = None
            
            if has_ground_truth:
                y_true = df_raw["target_class"].astype(str).tolist()
                
                # Check for any unseen target classes
                unique_trues = set(y_true)
                valid_classes = set(class_mapping.keys())
                unseen_trues = unique_trues - valid_classes
                if not unseen_trues:
                    # Compute classification report
                    report = classification_report(y_true, pred_labels, output_dict=True, zero_division=0)
                    metrics = {
                        "accuracy": report.get("accuracy", 0.0),
                        "f1_macro": report.get("macro avg", {}).get("f1-score", 0.0),
                        "precision_macro": report.get("macro avg", {}).get("precision", 0.0),
                        "recall_macro": report.get("macro avg", {}).get("recall", 0.0),
                        "details": report
                    }
                    
                    # Generate base64 Confusion Matrix Plot
                    try:
                        sorted_labels = sorted(list(class_mapping.keys()))
                        confusion_matrix_img = get_confusion_matrix_base64(y_true, pred_labels, sorted_labels)
                    except Exception as plt_err:
                        logger.error(f"Error plotting confusion matrix: {plt_err}")
                else:
                    logger.warning(f"Uploaded ground truth contains unseen categories: {unseen_trues}. Metrics skipped.")
            
            # Return preview of predictions (first 10 rows)
            preview_rows = df_output.head(10).to_dict(orient="records")
            
            return jsonify({
                "predictions_count": len(pred_labels),
                "preview": preview_rows,
                "csv_data": csv_data,
                "has_ground_truth": has_ground_truth,
                "metrics": metrics,
                "confusion_matrix_b64": confusion_matrix_img
            })
            
        else:
            return jsonify({"error": f"Invalid input_type: {input_type}"}), 400
            
    except Exception as e:
        logger.exception("Unexpected error during inference")
        return jsonify({"error": f"An internal server error occurred during prediction: {str(e)}"}), 500


if __name__ == "__main__":
    # In production, run on localhost and limit to loopback interface for local host safety
    app.run(host="127.0.0.1", port=5000, debug=True)
