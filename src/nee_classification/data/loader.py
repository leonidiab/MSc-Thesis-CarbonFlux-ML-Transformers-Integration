"""Low-level data loading for per-site CSV datasets."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_sites(
    base_dir: Path,
    sites: list[str],
    dataset_filename: str = "full_dataset.csv",
) -> pd.DataFrame:
    """Load and concatenate CSV datasets from multiple sites.

    Each site is expected to have its data at
    ``<base_dir>/<site>/<dataset_filename>``.  A ``_source_site`` column is
    added to every row to record which site the observation originated from.

    Parameters
    ----------
    base_dir:
        Root directory containing per-site sub-directories.
    sites:
        List of site names (must match sub-directory names).
    dataset_filename:
        Name of the CSV file inside each site directory.

    Returns
    -------
    pd.DataFrame
        Concatenated dataframe with an extra ``_source_site`` column.

    Raises
    ------
    FileNotFoundError
        If **none** of the expected CSV files exist.
    """
    base_dir = Path(base_dir)
    frames: list[pd.DataFrame] = []

    for site in sites:
        csv_path = base_dir / site / dataset_filename
        if not csv_path.is_file():
            logger.warning("CSV not found for site '%s': %s", site, csv_path)
            continue

        df = pd.read_csv(csv_path)
        df["_source_site"] = site
        frames.append(df)
        logger.info(
            "Loaded site '%s': %d rows, %d columns from %s",
            site,
            len(df),
            df.shape[1],
            csv_path,
        )

    if not frames:
        raise FileNotFoundError(
            f"No data found for any of the requested sites {sites} "
            f"under base directory '{base_dir}'"
        )

    combined = pd.concat(frames, ignore_index=True)
    logger.info(
        "Combined dataset: %d rows, %d columns from %d site(s)",
        len(combined),
        combined.shape[1],
        len(frames),
    )
    return combined
