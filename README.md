# NEE Cross-Site Classification - MSc Thesis _Carbon Flux Behavior Classification Integrating Transformers and Classical Machine Learning Methods_

Carbon flux (NEE) classification using tree-based and transformer models with cross-site test across Australian flux tower sites.

## Quick Start

### Installation

```bash
pip install -e ".[dev]"
```

### Training

Train a model with a specific holdout site:

```bash
# Tree-based models (XGBoost, LightGBM, RandomForest, etc.)
python scripts/train.py --config configs/cross_site/holdout_wombat.yaml --model trees

# FT-Transformer
python scripts/train.py --config configs/cross_site/holdout_wombat.yaml --model ft_transformer

# TabTransformer
python scripts/train.py --config configs/cross_site/holdout_wombat.yaml --model tabtransformer
```

### Cross-Site Testing

Test a trained model on the holdout site:

```bash
python scripts/test_cross_site.py \
    --model-dir outputs/holdout_wombat/trees/f1_macro/with_BOM/run_01 \
    --test-site wombat
```

### Interactive Testing Dashboard

You can run the web-based interactive testing dashboard to visually test your best saved models with new datasets:

```bash
python scripts/run_app.py
```

This will automatically start a secure local Flask server at `http://127.0.0.1:5000` and open the dashboard in your browser. Key features:
* **Model Selection Hub**: Automatically scans and displays metadata for all models in `outputs/`.
* **Batch Prediction (CSV)**: Drag and drop datasets, preview classifications, and download tagged results. If ground truth is included, it generates validation metrics and an interactive confusion matrix.
* **Real-time Form**: Test custom parameter values using input fields and sliders with class confidence probability meters.

For more details about the UI architecture, API REST endpoints, and security/preprocessing safeguards, refer to the [Complete User Guide](file:///C:/Users/leoni/Documents/Github/MSc-Thesis-CarbonFlux-ML-Transformers-Integration-DEV/documentation/USER_GUIDE.md#4-interactive-testing-dashboard-web-ui).

### CLI Options

```bash
python scripts/train.py --help
python scripts/test_cross_site.py --help
```

## Reproducing Experiments

Run all 15 experiments (3 model types × 5 holdout sites):

```bash
for site in cumberland robson_creek_queensland tumbarumba whroo wombat; do
    for model in trees ft_transformer tabtransformer; do
        python scripts/train.py \
            --config configs/cross_site/holdout_${site}.yaml \
            --model ${model}
    done
done
```

## Project Structure

```
├── configs/                    # YAML experiment configurations
│   ├── base.yaml               # Shared defaults
│   ├── cross_site/             # Per-holdout-site configs
│   └── models/                 # Per-model-type configs
├── data/processed/             # Preprocessed site datasets
│   ├── cumberland/
│   ├── robson_creek_queensland/
│   ├── tumbarumba/
│   ├── whroo/
│   └── wombat/
├── src/nee_classification/     # Main Python package
│   ├── config/                 # Pydantic v2 config schemas
│   ├── data/                   # Data loading & preprocessing
│   ├── models/                 # Trainer classes (trees, FT-T, TabT)
│   ├── evaluation/             # Metrics & report generation
│   ├── tuning/                 # Feature selection
│   ├── runner/                 # Worker & orchestrator
│   ├── artifacts/              # Model persistence
│   └── utils/                  # Seeds, system detection
├── scripts/                    # CLI entry points
│   ├── train.py
│   └── test_cross_site.py
├── tests/                      # pytest test suite
├── outputs/                    # Training results (gitignored)
└── pyproject.toml              # PEP 517/518 build config
```

## Data

Five Australian flux tower sites with 28 features each:

| Site | Rows | Description |
|------|------|-------------|
| Cumberland | 138 | Cumberland Plain |
| Robson Creek Queensland | 46 | Tropical rainforest |
| Tumbarumba | 642 | Wet sclerophyll forest |
| Whroo | 184 | Dry woodland |
| Wombat | 230 | Temperate forest |

**Features:** MODIS remote sensing (Fpar, LAI, GPP, ET, LE, LST, surface reflectance), BOM meteorological data (temperature, rainfall), and temporal features (year, month, week_of_year, day_of_year).

**Target:** Binary classification (`NS` = Non-Carbon sink, `S` = Carbon sink).

## Experimental Design

**Cross-site testing:** For each experiment, 4 sites are used for training and validation, and the remaining site is held out for independent testing. This evaluates the model's ability to generalise to unseen geographical locations.

**Optimisation:** Each experiment runs N independent Optuna trials across M metrics, with and without BOM features, producing a comprehensive comparison.

## Data

This project uses data from the **[FLUXNET2015 Dataset](https://fluxnet.org/)**.

**License:** [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)
**Citation:** Pastorello, G., Trotta, C., Canfora, E. *et al.* The FLUXNET2015 dataset and the ONEFlux processing pipeline for eddy covariance data. *Sci Data* **7**, 225 (2020). [https://doi.org/10.1038/s41597-020-0534-3](https://doi.org/10.1038/s41597-020-0534-3)

**Additional Sources:**
* **MODIS:** Courtesy of the NASA EOSDIS Land Processes Distributed Active Archive Center (LP DAAC).
* **[Bureau of Meteorology](https://www.bom.gov.au/) (BoM):** © Commonwealth of Australia, Bureau of Meteorology. Licensed under CC BY 3.0 AU.

## License

MIT
