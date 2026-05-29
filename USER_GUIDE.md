# NEE Classification Project - User Guide

This guide provides comprehensive instructions on how to configure, train, and test models within the NEE (Net Ecosystem Exchange) Classification framework. The project supports both traditional tree-based models (via Scikit-Learn, XGBoost, LightGBM) and deep learning models (FT-Transformer and TabTransformer via PyTorch Tabular).

## 1. Project Structure overview

- **`configs/`**: Contains YAML configuration files defining experiments (e.g., datasets to use, holdout sites, model-specific hyperparameters).
- **`data/processed/`**: The expected location for your processed datasets. Data should be organized by site, e.g., `data/processed/wombat/full_dataset.csv`.
- **`outputs/`**: Directory where training artifacts, models, and test results are saved.
- **`scripts/`**: Contains the main entry-point scripts:
  - `train.py`: For launching training and hyperparameter optimization experiments.
  - `test_cross_site.py`: For evaluating trained models on unseen (holdout) sites.
- **`src/nee_classification/`**: The core package containing data processing, model definitions, and evaluation logic.

## 2. Configuration (YAML Files)

Experiments are defined using YAML configuration files. These files specify the data splits, resource allocations, and model-specific parameters.

Example configuration (`configs/cross_site/holdout_wombat.yaml`):

```yaml
name: "cross_site_holdout_wombat"
model_type: "trees" # Default model type, can be overridden via CLI
n_runs: 10
n_trials: 600

data:
  all_sites: ["tumbarumba", "cumberland", "whroo", "wombat", "robson_creek_queensland"]
  train_sites: ["tumbarumba", "cumberland", "whroo", "robson_creek_queensland"]
  holdout_site: "wombat"
  target_col: "target_class"
  categorical_cols: ["year", "month", "week_of_year"]

resources:
  max_workers: "auto"
  force_cpu: false
```

You can use the `_inherit` key to create base configurations and extend them for specific sites to avoid duplication.

## 3. Training Models

Use the `scripts/train.py` script to run experiments. The script orchestrates data loading, parallel hyperparameter optimization (using Optuna), final model training, and artifact saving.

### Basic Usage

To train a model, you must provide a configuration file and specify the model type (`trees`, `ft_transformer`, or `tabtransformer`).

**Train Tree Models (XGBoost, LightGBM, Random Forest, etc.):**
```bash
python scripts/train.py --config configs/cross_site/holdout_wombat.yaml --model trees
```

**Train Deep Learning Models (FT-Transformer or TabTransformer):**
```bash
python scripts/train.py --config configs/cross_site/holdout_wombat.yaml --model ft_transformer
python scripts/train.py --config configs/cross_site/holdout_wombat.yaml --model tabtransformer
```

### CLI Overrides

You can override several configuration parameters directly from the command line, which is useful for quick testing or debugging:

- `--n-runs <int>`: Override the number of independent training runs.
- `--n-trials <int>`: Override the number of Optuna hyperparameter optimization trials per run.
- `--max-workers <int>`: Manually set the number of parallel workers.
- `--force-cpu`: Force training on CPU even if a CUDA GPU is available.
- `--max_epochs_optuna <int>`: (Transformers only) Max epochs during Optuna trials.
- `--max_epochs_final <int>`: (Transformers only) Max epochs for training the final model.
- `--early_stopping_patience <int>`: (Transformers only) Patience for early stopping.

**Example: Fast test run for a transformer model:**
```bash
python scripts/train.py --config configs/cross_site/holdout_wombat.yaml --model tabtransformer --n-runs 1 --n-trials 2 --max_epochs_optuna 5 --max_epochs_final 5
```

### Training Outputs

After training completes, the best model overall (across all optimized metrics) is saved to the `outputs/` directory.

For example, if you ran the `wombat` holdout experiment for `trees`, the artifacts will be saved in:
`outputs/holdout_wombat/best_trees_final/`

This directory will contain:
- `model.joblib` (for trees) or `model_final/` directory (for transformers): The actual trained model.
- `info.json`: Metadata about the run, best hyperparameters, and selected features.
- `best_model_summary.txt`: A quick human-readable summary of the experiment.
- `training_report.xlsx`: Excel file with detailed metrics for the train and validation splits.
- `confusion_matrix_val.png`: Confusion matrix plot for the validation set.
- Various `.joblib` files (e.g., scalers, categorical encoders, target encoders).

## 4. Cross-Site Testing

Once a model is trained and saved, you can evaluate its generalization performance on a holdout site using `scripts/test_cross_site.py`. This script handles data loading, necessary preprocessing (scaling, encoding) using the saved training artifacts, and evaluation.

### Basic Usage

You need to provide the directory containing the saved model and the name of the site to test against.

**Test a Tree model:**
```bash
python scripts/test_cross_site.py --model-dir outputs/holdout_wombat/best_trees_final --test-site wombat
```

**Test an FT-Transformer model:**
```bash
python scripts/test_cross_site.py --model-dir outputs/holdout_wombat/best_ft_transformer_final --test-site wombat
```

**Test a TabTransformer model:**
```bash
python scripts/test_cross_site.py --model-dir outputs/holdout_wombat/best_tabtransformer_final --test-site wombat
```

### Optional Arguments

- `--data-dir <path>`: Base directory for the processed data (defaults to `data/processed`).
- `--output-dir <path>`: Where to save the test results. By default, it creates a `test_results/` folder inside the provided `--model-dir`.

### Test Outputs

The test script prints a standard classification report to the console and generates the following files in the output directory:

- `classification_report_<site>.json`: The complete classification metrics in JSON format.
- `confusion_matrix_<site>.png`: A visualization of the confusion matrix for the test site.

## 5. Troubleshooting & Resource Management

- **CUDA Out of Memory (OOM):** Tree models (specifically XGBoost) and Transformer models can consume significant GPU VRAM. The orchestration script (`scripts/train.py`) includes safety limits:
  - If GPU is detected, `max_workers` for tree models is automatically capped at `2`.
  - For Transformer models, `max_workers` is automatically set to `1`.
  - If you encounter OOM errors, ensure no other heavy processes are using the GPU, or use the `--force-cpu` flag.
- **Missing `model_final/` directory:** For Transformer models, ensure your training script completed successfully and did not terminate early. The model is saved at the very end of the final training phase.