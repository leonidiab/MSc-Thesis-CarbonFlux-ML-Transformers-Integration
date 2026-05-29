"""Report generation utilities: confusion matrices, Excel, and JSON reports."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # noqa: E402 — must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

logger = logging.getLogger(__name__)


def save_classification_report(report_dict: dict, path: Path) -> None:
    """Persist a classification report dict as a JSON file.

    Parameters
    ----------
    report_dict : dict
        Output of ``classification_report(output_dict=True)``.
    path : Path
        Destination JSON file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report_dict, fh, indent=4, default=str)
    logger.info("Classification report saved to %s", path)


def save_confusion_matrix_plot(
    y_true,
    y_pred,
    path: Path,
    *,
    title: str = "",
    cmap: str = "Blues",
) -> None:
    """Save a confusion matrix visualisation as PNG.

    Parameters
    ----------
    y_true, y_pred : array-like
        True and predicted labels.
    path : Path
        Destination PNG file.
    title : str
        Optional title for the plot.
    cmap : str
        Matplotlib colour map name.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, cmap=cmap)
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", path)


def save_excel_report(
    output_path: Path,
    features: list[str],
    hyperparams: dict,
    metrics_data: list[dict],
    *,
    early_stop_epoch: int | str = "N/A",
) -> None:
    """Create a multi-sheet Excel report summarising a training run.

    Sheets
    ------
    - **Selected_Features** — ordered list of features used.
    - **Hyperparameters** — final hyper-parameter values.
    - **Final_Metrics** — per-class and aggregate metrics for each split.

    Parameters
    ----------
    output_path : Path
        Destination ``.xlsx`` file.
    features : list[str]
        Ordered feature names.
    hyperparams : dict
        Best hyper-parameter dict.
    metrics_data : list[dict]
        Each dict must have keys ``split`` (str) and ``report`` (classification
        report dict).
    early_stop_epoch : int or str
        Epoch at which early stopping fired, or ``"N/A"``.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        # Sheet 1: features
        pd.DataFrame(features, columns=["Selected_Feature"]).to_excel(
            writer, sheet_name="Selected_Features", index=False
        )

        # Sheet 2: hyperparameters
        hp = dict(hyperparams)
        hp["early_stop_epoch"] = early_stop_epoch
        pd.DataFrame.from_dict(hp, orient="index", columns=["Value"]).to_excel(
            writer, sheet_name="Hyperparameters"
        )

        # Sheet 3: metrics per split
        all_metrics: list[pd.DataFrame] = []
        for entry in metrics_data:
            df_report = (
                pd.DataFrame(entry["report"])
                .transpose()
                .reset_index()
                .rename(columns={"index": "Class"})
            )
            df_report["Split"] = entry["split"]
            all_metrics.append(df_report)

        if all_metrics:
            pd.concat(all_metrics, ignore_index=True).to_excel(
                writer, sheet_name="Final_Metrics", index=False
            )

    logger.info("Excel report saved to %s", output_path)
