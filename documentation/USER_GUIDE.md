# NEE Classification Project - Complete User Guide

This guide provides comprehensive instructions on how to configure, train, test, and interactively evaluate models within the NEE (Net Ecosystem Exchange) Classification framework.

---

## 1. Project Structure Overview

* **`configs/`**: YAML configuration files defining experiments (datasets, holdout sites, model hyperparameters).
* **`data/processed/`**: Processed datasets by site (e.g., `data/processed/wombat/full_dataset.csv`).
* **`documentation/`**: Comprehensive project manuals and reports.
* **`outputs/`**: Saved models, preprocessing encoders, and reports.
* **`scripts/`**: Orchestration and entry-point scripts:
  * `train.py`: Starts training and Optuna hyperparameter tuning.
  * `test_cross_site.py`: Tests models on holdout site datasets via CLI.
  * `run_app.py`: Launches the interactive model testing web dashboard.
* **`src/nee_classification/`**: Core library:
  * `data/`: Data loaders and preprocessors.
  * `models/`: Trainer classes (trees and deep learning).
  * `web/`: Backend Flask server, static files, and templates.

---

## 2. Configuration (YAML Files)

Experiments are defined using YAML configurations. They specify the train/test splits, resource allocations, and feature options.

Example base configuration (`configs/base.yaml`):
```yaml
name: base_experiment
random_seed_base: 42
n_runs: 10
n_trials: 600
optimization_metrics: [accuracy, f1_macro]
primary_metric: f1_macro
data:
  base_dir: data/processed
  target_col: target_class
  features_to_drop_always: [NEE_VUT_REF, TIMESTAMP]
```

---

## 3. Training Models

Use `scripts/train.py` to run experiments. The script manages data loading, categorical/numerical preprocessing, Optuna tuning, and copy-saves the best model to the `outputs/` folder.

**Train Tree Models (XGBoost, LightGBM, Random Forest):**
```bash
python scripts/train.py --config configs/cross_site/holdout_wombat.yaml --model trees
```

**Train Deep Learning Models (FT-Transformer or TabTransformer):**
```bash
python scripts/train.py --config configs/cross_site/holdout_wombat.yaml --model ft_transformer
python scripts/train.py --config configs/cross_site/holdout_wombat.yaml --model tabtransformer
```

*Note: For Deep Learning models on machines with limited RAM, it is recommended to pass `--max-workers 1` to prevent Out-Of-Memory exceptions.*

---

## 4. Interactive Testing Dashboard (Web UI)

The project includes an interactive web-based graphical user interface for model exploration and inference. This tool allows users to select any trained model residing in the `outputs/` directory, perform predictions on individual custom instances using sliders tailored to each feature, or upload entire CSV files for batch classification.

### 4.1 File Structure

The web interface is structured using a classic Flask (Backend) + HTML/CSS/JS (Frontend) architecture:
* **Execution Script**: [scripts/run_app.py](file:///C:/Users/leoni/Documents/Github/MSc-Thesis-CarbonFlux-ML-Transformers-Integration-DEV/scripts/run_app.py)
* **Flask Backend**: [src/nee_classification/web/server.py](file:///C:/Users/leoni/Documents/Github/MSc-Thesis-CarbonFlux-ML-Transformers-Integration-DEV/src/nee_classification/web/server.py)
* **HTML Template**: [src/nee_classification/web/templates/index.html](file:///C:/Users/leoni/Documents/Github/MSc-Thesis-CarbonFlux-ML-Transformers-Integration-DEV/src/nee_classification/web/templates/index.html)
* **JS Logic (Client-side)**: [src/nee_classification/web/static/app.js](file:///C:/Users/leoni/Documents/Github/MSc-Thesis-CarbonFlux-ML-Transformers-Integration-DEV/src/nee_classification/web/static/app.js)
* **CSS Styling (Premium Theme)**: [src/nee_classification/web/static/style.css](file:///C:/Users/leoni/Documents/Github/MSc-Thesis-CarbonFlux-ML-Transformers-Integration-DEV/src/nee_classification/web/static/style.css)

### 4.2 Installation Requirements

To run the dashboard, ensure the required dependencies are installed in your virtual environment:
```bash
pip install flask pandas scikit-learn joblib xgboost matplotlib seaborn
# Optional (required to load FT-Transformer or TabTransformer deep learning models)
pip install pytorch-tabular pytorch-lightning
```

### 4.3 Launching the App

Execute the following command in the project root:
```bash
python scripts/run_app.py
```
This script starts the Flask server at `http://127.0.0.1:5000` and automatically opens the application in your default web browser. If no trained models are detected in the `outputs/` directory, the launcher displays a warning prompting you to train a model first using `scripts/train.py`.

### 4.4 Dashboard Features

The dashboard consists of three main sections organized in an "Eco-Friendly Premium" design (mint green/forest green accents with translucent glassmorphic card panels):

1. **Model Selection Hub**:
   - Scans the `outputs/` directory dynamically when the page is loaded.
   - Displays available models as cards containing: simplified model type (`XGBoost`, `FT-Transformer`, or `TabTransformer`), the holdout validation site, BoM feature usage, primary optimization metric, and best validation score.
   - Click **"Load Model"** to instantly load a model into the server's memory.

2. **Batch Prediction (CSV)**:
   - Supports drag-and-drop file uploading or file browsing.
   - **Interactive Preview Table**: Renders the first 10 rows of data side-by-side with predicted classification categories.
   - **Download Labeled Dataset**: Exports the full dataset containing the original columns plus prediction outputs (`predicted_class` marked as `"S"` or `"NS"`) and class probabilities (`probability_S` and `probability_NS`).
   - **Metrics & Confusion Matrix**: If the uploaded CSV contains a `target_class` column with true labels, the backend evaluates classification metrics (Accuracy, F1-macro, Precision, Recall) and plots a customized confusion matrix heatmap embedded directly in the UI.

3. **Real-time Single Instance Entry**:
   - Dynamically constructs a form based on features expected by the active model.
   - **Numeric Inputs & Sliders**: Automatically set boundaries matching the minimum/maximum bounds seen in training data to prevent physically impossible entries.
   - **Categorical Dropdowns**: Provide structured selectors for time variables like Year, Month, Day of Year, and Week of Year.
   - **Instant Prediction**: Submits inputs via JSON API, displaying a classification badge (**Sink (S)** vs **No Sink (NS)**) and live progress meters for class probabilities.

---

## 5. REST API Endpoints (Backend)

The Flask server hosts two primary endpoints for client communications:

### 5.1 List Models
* **Endpoint**: `/api/models`
* **Method**: `GET`
* **Description**: Scans the `outputs/` folder for `info.json` files and returns a structured array of models that are ready for inference (i.e. containing `model.joblib` or a `model_final` folder).
* **Sample Response**:
  ```json
  [
    {
      "id": "holdout_wombat/trees/f1_macro/with_BOM/run_01",
      "model_type": "trees",
      "metric_optimized": "f1_macro",
      "best_val_score": 0.842,
      "use_bom": true,
      "features": ["Fpar", "LAI", "Ta_F", "Precip_F"],
      "holdout_site": "wombat",
      "class_mapping": {"NS": 0, "S": 1}
    }
  ]
  ```

### 5.2 Execute Inference
* **Endpoint**: `/api/predict`
* **Method**: `POST`
* **Request Content-Type**: `multipart/form-data`
* **Parameters**:
  - `model_id` (string, required): Relative model path (e.g. `holdout_wombat/trees/...`).
  - `input_type` (string, required): Either `"single"` or `"file"`.
  - `data` (stringified JSON, required for `input_type="single"`): Dictionary mapping feature names to values.
  - `file` (CSV file, required for `input_type="file"`): Input CSV file.
* **Sample Responses**:
  - **Single Instance**:
    ```json
    {
      "prediction": "S",
      "probabilities": {
        "NS": 0.1245,
        "S": 0.8755
      }
    }
    ```
  - **Batch (CSV)**:
    ```json
    {
      "predictions_count": 230,
      "preview": [ ... ],
      "csv_data": "Fpar,LAI,predicted_class,...\n0.6,1.2,S,...\n",
      "has_ground_truth": true,
      "metrics": {
        "accuracy": 0.891,
        "f1_macro": 0.887,
        "precision_macro": 0.89,
        "recall_macro": 0.885,
        "details": { ... }
      },
      "confusion_matrix_b64": "iVBORw0KGgoAAA..."
    }
    ```

---

## 6. Preprocessing & Security Controls

Exposing ML models via a web interface requires rigorous controls, implemented in [src/nee_classification/web/server.py](file:///C:/Users/leoni/Documents/Github/MSc-Thesis-CarbonFlux-ML-Transformers-Integration-DEV/src/nee_classification/web/server.py):

* **Path Traversal Protection**: Client-supplied model directories are resolved into absolute paths and verified to stay within `outputs/`. Requests attempting to escape (e.g. via `..`) trigger a `PermissionError` and return `403 Forbidden`.
* **Safe Deserialization**: The application only loads models stored locally. It never accepts direct model binaries (such as `.joblib` files) uploaded from external clients.
* **Feature Column Pad Guard**: If user inputs (forms or CSVs) lack columns expected by encoders or scaling functions, the backend fills those fields with default values (`0` or `0.0`) and aligns column orders transparently before invoking the inference engine.
* **NaN Value Handling**: During CSV file processing, missing numerical feature fields are automatically imputed with column medians to prevent standardizer or neural network crashes.
* **HTTP Security Headers**: Secure headers (CSP, X-Frame-Options, X-XSS-Protection, X-Content-Type-Options) are applied to all HTTP responses.
