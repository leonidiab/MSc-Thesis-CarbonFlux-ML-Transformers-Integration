# Project Analysis & Web App Security Report

This report provides a comprehensive review of the **Carbon Flux (NEE) Classification** project, detailing its architecture, ML pipelines, UI/UX design decisions, and the security blueprints implemented for safe local or network web deployment.

---

## 1. Project Architecture & Pipeline Design

The project classifies Net Ecosystem Exchange (NEE) targets into **S** (Carbon Sink) or **NS** (Not Sink / Source) using traditional machine learning ensembles (via Scikit-Learn, XGBoost, and LightGBM) and deep learning tabular models (FT-Transformer and TabTransformer via PyTorch Tabular).

### 1.1 Data Preprocessing Flow
1. **Cleaning**: Provenance columns and unneeded features (such as `TIMESTAMP` and the direct target indicator `NEE_VUT_REF`) are dropped.
2. **Encoding**: Categorical columns (`year`, `month`, `week_of_year`) are encoded via `LabelEncoder`.
3. **Scaling**: Numerical features are standardized using `StandardScaler` to ensure optimal gradient updates for deep learning networks.
4. **Serialization**: Encoder and scaler states are saved as serializable `.joblib` files to ensure reproducible transforms during batch or real-time inference.

---

## 2. Interactive Testing Dashboard UI/UX

To allow anyone who downloads the repository to immediately test the trained models, we developed a lightweight, single-page dashboard featuring:
* **Interactive Model Hub**: Grid of model cards scanned dynamically from the `outputs/` folder. It lists model type, best validation score, holdout site, and feature configurations.
* **Dual-Input Paradigm**:
  * **Batch CSV Upload**: Features a file drop zone. Uploaded datasets are parsed, preprocessed, labeled, and prepared for download as standard CSVs. If the file contains actual class labels (`target_class`), the app automatically generates performance statistics and an interactive Confusion Matrix heatmap.
  * **Real-time Feature Entry**: Dynamically renders input elements matching the active model's features. Uses custom sliders with boundaries aligned with the biological and physical limits of each column (e.g. Leaf Area Index limited to `0-8`, temperature to `-15°C to 50°C`).
* **Eco-Theme Light Mode**: Styled using fresh light green tones (sages and mints) with deep green headings, backdrop filter blurs (glassmorphism), and animated layouts.

---

## 3. Web & App Interface Security Blueprint

Exposing python-based machine learning pipelines to web endpoints introduces critical vulnerabilities. The following layers were implemented to protect the host environment and web clients:

### 3.1 Path Traversal Protection
* **Vulnerability**: Path traversal (`../../etc/passwd`) allows attackers to read or overwrite system files by manipulating model IDs or file parameters.
* **Safeguard**: The backend implements `safe_resolve_path()`. It forces paths to resolve against the project's absolute base `outputs/` directory and explicitly checks that the target path begins exactly with the resolved path of the outputs folder. Access is blocked if any traversal attempts are detected.

### 3.2 Secure Model Serialization
* **Vulnerability**: Python's `pickle` and `joblib` serializers are inherently unsafe. Deserializing an untrusted model file can trigger Remote Code Execution (RCE) by executing malicious code hidden within the file payload.
* **Safeguard**: The web backend **never** exposes endpoints to upload new model files (.joblib or PyTorch folders) via HTTP. The server only loads models from local directories that reside inside the project's own `outputs/` folder, ensuring only internally trained models are executed.

### 3.3 Dynamic Schema Verification (Robust Inference)
* **Vulnerability**: Mismatches in expected columns (due to out-of-order lists or missing temporal categories) can trigger server-side errors, crash threads, or leak system tracebacks.
* **Safeguard**: 
  - The server dynamically inspects standardizer and encoder parameters (`scaler.feature_names_in_` and model configs).
  - Any missing columns in user-supplied entries are padded with default values (such as `0` or `0.0`), and all data columns are re-ordered to match the exact sequence expected by the model.

### 3.4 Security Headers & CSP
* **Safeguard**: The Flask backend automatically injects the following security headers into every HTTP response:
  * `Content-Security-Policy (CSP)`: Disallows arbitrary scripts, styles, or frames from being executed or embedded.
  * `X-Content-Type-Options: nosniff`: Prevents MIME type sniffing.
  * `X-Frame-Options: DENY`: Blocks clickjacking.
  * `X-XSS-Protection: 1; mode=block`: Enables cross-site scripting blocking.
