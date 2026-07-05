// ====================================================================
// State management
// ====================================================================
let selectedModelId = null;
let selectedInputMethod = null;   // 'csv' | 'manual'
let currentModels = [];
let batchCsvData = null;

// Helper to clean model names for display
function getModelDisplayName(modelType) {
    const typeLower = modelType.toLowerCase();
    if (typeLower.includes("ft_transformer") || typeLower.includes("ft-transformer") || typeLower.includes("fttransformer")) {
        return "FT-Transformer";
    }
    if (typeLower.includes("tabtransformer")) {
        return "TabTransformer";
    }
    if (typeLower.includes("xgboost") || typeLower.includes("xgb")) {
        return "XGBoost";
    }
    return modelType.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

// ====================================================================
// DOM Elements
// ====================================================================

// Screens
const landingSection = document.getElementById("landing-section");
const setupSection = document.getElementById("setup-section");
const inferenceSection = document.getElementById("inference-section");

// Landing
const btnGetStarted = document.getElementById("btn-get-started");

// Setup — Navigation
const btnBackToLanding = document.getElementById("btn-back-to-landing");
const btnProceedInference = document.getElementById("btn-proceed-inference");
const setupHint = document.getElementById("setup-hint");

// Setup — Model grid
const modelsListEl = document.getElementById("models-list");
const btnRefreshModels = document.getElementById("btn-refresh-models");

// Setup — Input method
const methodCards = document.querySelectorAll(".input-method-card");

// Inference — Navigation
const btnBackToSetup = document.getElementById("btn-back-to-setup");

// Inference — Banner
const bannerModelType = document.getElementById("banner-model-type");
const bannerModelPath = document.getElementById("banner-model-path");

// Toast elements
const errorToast = document.getElementById("error-toast");
const errorToastMsg = document.getElementById("error-toast-msg");
const btnCloseToast = document.getElementById("btn-close-toast");

// Tabs Elements
const tabHeaders = document.querySelectorAll(".tab-header");
const tabContents = document.querySelectorAll(".tab-content");

// File Upload Elements
const csvDropZone = document.getElementById("csv-drop-zone");
const csvFileInput = document.getElementById("csv-file-input");
const fileDetailsPanel = document.getElementById("file-details-panel");
const selectedFileName = document.getElementById("selected-file-name");
const selectedFileSize = document.getElementById("selected-file-size");
const btnRunBatchInference = document.getElementById("btn-run-batch-inference");
const batchResultsPanel = document.getElementById("batch-results-panel");
const metricsDashboard = document.getElementById("metrics-dashboard");
const btnDownloadResults = document.getElementById("btn-download-results");
const previewTableHeader = document.getElementById("preview-table-header");
const previewTableBody = document.getElementById("preview-table-body");
const confusionMatrixContainer = document.getElementById("confusion-matrix-container");
const confusionMatrixImg = document.getElementById("confusion-matrix-img");

// Manual Form Elements
const manualFeaturesGrid = document.getElementById("manual-features-grid");
const manualInferenceForm = document.getElementById("manual-inference-form");
const manualResultCard = document.getElementById("manual-result-card");
const manualPredictionVal = document.getElementById("manual-prediction-val");
const manualProbMeters = document.getElementById("manual-prob-meters");

// ====================================================================
// Initialization
// ====================================================================
document.addEventListener("DOMContentLoaded", () => {
    loadModels();
    setupEventListeners();
});

// ====================================================================
// Screen Navigation
// ====================================================================
function showScreen(screenName) {
    // Hide all screens
    landingSection.classList.remove("active");
    setupSection.classList.remove("active");
    inferenceSection.classList.remove("active");

    // Show the requested screen
    switch (screenName) {
        case "landing":
            landingSection.classList.add("active");
            break;
        case "setup":
            setupSection.classList.add("active");
            break;
        case "inference":
            inferenceSection.classList.add("active");
            break;
    }

    // Scroll to top on screen change
    window.scrollTo({ top: 0, behavior: "smooth" });
}

// Update the "Proceed" button state based on current selections
function updateProceedState() {
    const canProceed = selectedModelId !== null && selectedInputMethod !== null;
    btnProceedInference.disabled = !canProceed;

    if (canProceed) {
        setupHint.textContent = "All set! Click above to proceed.";
        setupHint.style.color = "var(--primary)";
    } else {
        const parts = [];
        if (!selectedModelId) parts.push("a model");
        if (!selectedInputMethod) parts.push("an input method");
        setupHint.textContent = `Please select ${parts.join(" and ")} to continue.`;
        setupHint.style.color = "";
    }
}

// ====================================================================
// Toast notification
// ====================================================================
function showToast(message, type = "error") {
    errorToastMsg.textContent = message;
    errorToast.classList.remove("hidden");
    
    // Automatically close after 6 seconds
    setTimeout(() => {
        hideToast();
    }, 6000);
}

function hideToast() {
    errorToast.classList.add("hidden");
}

// ====================================================================
// Event Listeners Setup
// ====================================================================
function setupEventListeners() {
    // --- Screen navigation ---
    btnGetStarted.addEventListener("click", () => showScreen("setup"));
    btnBackToLanding.addEventListener("click", () => showScreen("landing"));
    btnBackToSetup.addEventListener("click", () => {
        showScreen("setup");
        // Reset inference visual state
        batchResultsPanel.classList.add("hidden");
        manualResultCard.classList.add("hidden");
    });

    btnProceedInference.addEventListener("click", () => {
        if (!selectedModelId || !selectedInputMethod) return;

        // Find the selected model object
        const model = currentModels.find(m => m.id === selectedModelId);
        if (!model) return;

        // Update inference banner
        const displayName = getModelDisplayName(model.model_type);
        bannerModelType.textContent = displayName;
        bannerModelPath.textContent = `outputs/${model.id}`;

        // Build manual inputs for the model
        renderManualInputs(model.features);

        // Activate the correct tab based on selected input method
        activateTab(selectedInputMethod === "csv" ? "tab-batch" : "tab-manual");

        // Reset output containers
        batchResultsPanel.classList.add("hidden");
        manualResultCard.classList.add("hidden");
        fileDetailsPanel.classList.add("hidden");

        // Navigate
        showScreen("inference");
    });

    // --- Input method selection ---
    methodCards.forEach(card => {
        card.addEventListener("click", () => {
            methodCards.forEach(c => c.classList.remove("selected"));
            card.classList.add("selected");
            selectedInputMethod = card.getAttribute("data-method");
            updateProceedState();
        });
    });

    // --- Model grid refresh ---
    btnRefreshModels.addEventListener("click", loadModels);

    // --- Toast ---
    btnCloseToast.addEventListener("click", hideToast);

    // --- Tab switcher logic ---
    tabHeaders.forEach(header => {
        header.addEventListener("click", () => {
            const targetTab = header.getAttribute("data-tab");
            activateTab(targetTab);
        });
    });

    // --- Drag and Drop ---
    csvDropZone.addEventListener("click", () => csvFileInput.click());
    
    csvDropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        csvDropZone.classList.add("drag-over");
    });
    
    csvDropZone.addEventListener("dragleave", () => {
        csvDropZone.classList.remove("drag-over");
    });
    
    csvDropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        csvDropZone.classList.remove("drag-over");
        if (e.dataTransfer.files.length > 0) {
            handleFileSelection(e.dataTransfer.files[0]);
        }
    });

    csvFileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFileSelection(e.target.files[0]);
        }
    });

    // --- Batch prediction execution ---
    btnRunBatchInference.addEventListener("click", runBatchInference);

    // --- Manual prediction execution ---
    manualInferenceForm.addEventListener("submit", runManualInference);

    // --- Results download action ---
    btnDownloadResults.addEventListener("click", downloadBatchResults);
}

// ====================================================================
// Tab Activation Helper
// ====================================================================
function activateTab(tabId) {
    tabHeaders.forEach(h => h.classList.remove("active"));
    tabContents.forEach(c => c.classList.remove("active"));

    const targetHeader = document.querySelector(`.tab-header[data-tab="${tabId}"]`);
    if (targetHeader) targetHeader.classList.add("active");
    const targetContent = document.getElementById(tabId);
    if (targetContent) targetContent.classList.add("active");
}

// ====================================================================
// Model Loading & Rendering
// ====================================================================
async function loadModels() {
    try {
        modelsListEl.innerHTML = `
            <div class="loading-state">
                <div class="spinner"></div>
                <p>Scanning outputs directory for saved models...</p>
            </div>
        `;
        
        const response = await fetch("/api/models");
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        currentModels = await response.json();
        
        // Sort models by performance: best validation score (descending) first, worst last
        currentModels.sort((a, b) => b.best_val_score - a.best_val_score);
        
        renderModels(currentModels);
    } catch (err) {
        showToast("Could not communicate with the backend. Ensure the server script is running.");
        modelsListEl.innerHTML = `
            <div class="loading-state">
                <p class="error-msg">⚠️ Communication Error: ${err.message}</p>
            </div>
        `;
    }
}

// Render dynamic model cards
function renderModels(models) {
    if (models.length === 0) {
        modelsListEl.innerHTML = `
            <div class="loading-state" style="grid-column: 1 / -1;">
                <p>📂 No models found inside the outputs/ directory.</p>
                <p style="font-size: 0.85rem; margin-top: 0.5rem; text-align: center;">
                    Please run the training pipeline first using: <br>
                    <code style="background: rgba(0,0,0,0.3); padding: 2px 6px; border-radius: 4px;">
                        python scripts/train.py --config configs/cross_site/holdout_wombat.yaml --model trees
                    </code>
                </p>
            </div>
        `;
        return;
    }

    modelsListEl.innerHTML = "";
    models.forEach(model => {
        const card = document.createElement("div");
        card.className = "model-card";
        if (selectedModelId === model.id) {
            card.classList.add("selected");
        }
        
        const displayName = getModelDisplayName(model.model_type);
        
        card.innerHTML = `
            <div class="model-title-row">
                <span class="model-name">${displayName}</span>
            </div>
            
            <div class="model-details">
                <div class="detail-item">
                    <span class="detail-label">Optimized Metric:</span>
                    <span class="detail-val">${model.metric_optimized}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Holdout Site:</span>
                    <span class="detail-val">${model.holdout_site}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">BOM Weather Data:</span>
                    <span class="detail-val">${model.use_bom ? 'Enabled' : 'Disabled'}</span>
                </div>
            </div>
            
            <div class="model-footer-score">
                <span class="score-label">Best Score</span>
                <span class="score-val">${model.best_val_score.toFixed(4)}</span>
            </div>
        `;
        
        card.addEventListener("click", () => selectModel(model));
        modelsListEl.appendChild(card);
    });
}

// Select active model (in setup screen only — does NOT navigate)
function selectModel(model) {
    selectedModelId = model.id;
    
    // Highlight active card
    const cards = modelsListEl.querySelectorAll(".model-card");
    cards.forEach((card, idx) => {
        if (currentModels[idx].id === model.id) {
            card.classList.add("selected");
        } else {
            card.classList.remove("selected");
        }
    });

    updateProceedState();
}

// ====================================================================
// Manual Feature Inputs Builder
// ====================================================================
function renderManualInputs(features) {
    manualFeaturesGrid.innerHTML = "";
    
    // Pre-sort features (temporal/categorical last or first)
    features.forEach(featName => {
        const group = document.createElement("div");
        group.className = "form-group";
        
        let min = 0;
        let max = 100;
        let val = 10;
        let step = 1;
        let isSelect = false;
        let selectOptions = [];

        // Check features category by names and map boundaries
        if (featName.includes("temp") || featName.includes("LST")) {
            min = -15;
            max = 50;
            val = 20;
            step = 0.1;
        } else if (featName.includes("Fpar") || featName.includes("Gpp")) {
            min = 0;
            max = 5;
            val = 0.5;
            step = 0.01;
        } else if (featName.includes("Lai")) {
            min = 0;
            max = 8;
            val = 3;
            step = 0.1;
        } else if (featName.includes("Rainfall")) {
            min = 0;
            max = 150;
            val = 0;
            step = 0.1;
        } else if (featName.includes("ET") || featName.includes("LE") || featName.includes("PLE") || featName.includes("PET")) {
            min = 0;
            max = 30000000;
            val = 5000000;
            step = 1000;
        } else if (featName.includes("refl")) {
            min = 0;
            max = 1000;
            val = 150;
            step = 0.1;
        } else if (featName === "year") {
            isSelect = true;
            for (let y = 2000; y <= 2030; y++) selectOptions.push(y);
            val = 2026;
        } else if (featName === "month") {
            isSelect = true;
            for (let m = 1; m <= 12; m++) selectOptions.push(m);
            val = 1;
        } else if (featName === "week_of_year") {
            isSelect = true;
            for (let w = 1; w <= 53; w++) selectOptions.push(w);
            val = 1;
        } else if (featName === "day_of_year") {
            isSelect = true;
            for (let d = 1; d <= 366; d++) selectOptions.push(d);
            val = 1;
        }

        // Render HTML for inputs
        if (isSelect) {
            group.innerHTML = `
                <label for="inp-${featName}">${featName}</label>
                <select name="${featName}" id="inp-${featName}">
                    ${selectOptions.map(opt => `<option value="${opt}" ${opt === val ? 'selected' : ''}>${opt}</option>`).join('')}
                </select>
            `;
        } else {
            // Check large numbers formatting
            const isLargeVal = max > 100000;
            
            group.innerHTML = `
                <label for="inp-${featName}">${featName}</label>
                <div class="slider-container">
                    <input type="range" name="${featName}" id="inp-${featName}" min="${min}" max="${max}" step="${step}" value="${val}">
                    <span class="slider-val" id="val-${featName}">${isLargeVal ? (val/1e6).toFixed(1) + 'M' : val}</span>
                </div>
            `;
            
            // Slider value visual update listener
            const slider = group.querySelector(`input[type="range"]`);
            const displaySpan = group.querySelector(`.slider-val`);
            slider.addEventListener("input", (e) => {
                const currentVal = parseFloat(e.target.value);
                displaySpan.textContent = isLargeVal ? (currentVal/1e6).toFixed(1) + 'M' : currentVal;
            });
        }
        
        manualFeaturesGrid.appendChild(group);
    });
}

// ====================================================================
// File Selection Handler
// ====================================================================
function handleFileSelection(file) {
    if (!file.name.endsWith(".csv")) {
        showToast("Invalid file type. Please upload a CSV file (.csv).");
        return;
    }
    
    selectedFileName.textContent = file.name;
    // Format file size
    const sizeKB = (file.size / 1024).toFixed(1);
    selectedFileSize.textContent = `${sizeKB} KB`;
    
    fileDetailsPanel.classList.remove("hidden");
    batchResultsPanel.classList.add("hidden");
}

// ====================================================================
// Batch Inference
// ====================================================================
async function runBatchInference() {
    const file = csvFileInput.files[0] || (csvDropZone.querySelector('.hidden-input') && csvDropZone.querySelector('.hidden-input').files[0]);
    if (!file) {
        showToast("Please choose or drop a CSV file first.");
        return;
    }

    const formData = new FormData();
    formData.append("model_id", selectedModelId);
    formData.append("input_type", "file");
    formData.append("file", file);

    try {
        btnRunBatchInference.disabled = true;
        btnRunBatchInference.textContent = "⚙️ Executing predictions...";
        
        const response = await fetch("/api/predict", {
            method: "POST",
            body: formData
        });
        
        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.error || `Server returned error status: ${response.status}`);
        }
        
        renderBatchResults(result);
    } catch (err) {
        showToast(err.message);
    } finally {
        btnRunBatchInference.disabled = false;
        btnRunBatchInference.textContent = "🚀 Run Batch Predictions";
    }
}

// Render batch results elements
function renderBatchResults(result) {
    batchCsvData = result.csv_data;
    
    // Render metrics if ground truth was present
    if (result.has_ground_truth && result.metrics) {
        document.getElementById("metric-accuracy").textContent = (result.metrics.accuracy * 100).toFixed(1) + "%";
        document.getElementById("metric-f1").textContent = result.metrics.f1_macro.toFixed(4);
        document.getElementById("metric-precision").textContent = result.metrics.precision_macro.toFixed(4);
        document.getElementById("metric-recall").textContent = result.metrics.recall_macro.toFixed(4);
        metricsDashboard.classList.remove("hidden");
    } else {
        metricsDashboard.classList.add("hidden");
    }

    // Render confusion matrix if available
    if (result.confusion_matrix_b64) {
        confusionMatrixImg.src = `data:image/png;base64,${result.confusion_matrix_b64}`;
        confusionMatrixContainer.classList.remove("hidden");
    } else {
        confusionMatrixContainer.classList.add("hidden");
    }

    // Render Predictions preview table
    if (result.preview && result.preview.length > 0) {
        // Headers
        const cols = Object.keys(result.preview[0]);
        previewTableHeader.innerHTML = cols.map(col => `<th>${col}</th>`).join("");
        
        // Rows
        previewTableBody.innerHTML = result.preview.map(row => {
            return `<tr>${cols.map(col => {
                const cellVal = row[col];
                if (col === "predicted_class" || col === "target_class") {
                    const isSink = cellVal === "S";
                    const cls = isSink ? "s-label" : "ns-label";
                    return `<td class="predicted-cell ${cls}">${cellVal}</td>`;
                }
                // Format numeric features
                if (typeof cellVal === "number" && !Number.isInteger(cellVal)) {
                    return `<td>${cellVal.toFixed(4)}</td>`;
                }
                return `<td>${cellVal}</td>`;
            }).join("")}</tr>`;
        }).join("");
    }

    batchResultsPanel.classList.remove("hidden");
}

// ====================================================================
// CSV Download
// ====================================================================
function downloadBatchResults() {
    if (!batchCsvData) return;
    
    const blob = new Blob([batchCsvData], { type: "text/csv;charset=utf-8;" });
    const link = document.createElement("a");
    const url = URL.createObjectURL(blob);
    
    link.setAttribute("href", url);
    link.setAttribute("download", `predictions_${selectedFileName.textContent || 'dataset'}`);
    link.style.visibility = 'hidden';
    
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// ====================================================================
// Manual Single-Instance Inference
// ====================================================================
async function runManualInference(e) {
    e.preventDefault();
    
    const formData = new FormData(manualInferenceForm);
    const dataObj = {};
    for (let [key, val] of formData.entries()) {
        dataObj[key] = parseFloat(val);
    }

    const reqData = new FormData();
    reqData.append("model_id", selectedModelId);
    reqData.append("input_type", "single");
    reqData.append("data", JSON.stringify(dataObj));

    try {
        const btn = document.getElementById("btn-run-manual-inference");
        btn.disabled = true;
        btn.textContent = "⚙️ Executing...";
        
        const response = await fetch("/api/predict", {
            method: "POST",
            body: reqData
        });
        
        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.error || `Server error status: ${response.status}`);
        }
        
        renderManualResult(result);
    } catch (err) {
        showToast(err.message);
    } finally {
        const btn = document.getElementById("btn-run-manual-inference");
        btn.disabled = false;
        btn.textContent = "🧪 Run Real-Time Prediction";
    }
}

// Display manual inference prediction badge and probabilities
function renderManualResult(result) {
    const isSink = result.prediction === "S";
    manualPredictionVal.textContent = isSink ? "Sink (S)" : "No Sink (NS)";
    
    manualPredictionVal.className = "prediction-badge";
    manualPredictionVal.classList.add(isSink ? "sink" : "no-sink");

    // Render progress bars for probabilities
    if (result.probabilities) {
        manualProbMeters.innerHTML = Object.entries(result.probabilities).map(([cls, prob]) => {
            const pct = (prob * 100).toFixed(1);
            const fillClass = cls === "S" ? "sink" : "no-sink";
            return `
                <div class="prob-meter-row">
                    <div class="prob-label-row">
                        <span>Class ${cls === 'S' ? 'Sink (S)' : 'Not Sink (NS)'}</span>
                        <span>${pct}%</span>
                    </div>
                    <div class="prob-track">
                        <div class="prob-fill ${fillClass}" style="width: ${pct}%"></div>
                    </div>
                </div>
            `;
        }).join("");
    } else {
        manualProbMeters.innerHTML = "";
    }
    
    manualResultCard.classList.remove("hidden");
    // Auto scroll to results
    manualResultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}
