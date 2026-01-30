"""Data loading utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_target_series(data_path: Path, hdf_key: str, target_col: str) -> pd.DataFrame:
    """Load HDF and select target column."""
    raw = pd.read_hdf(data_path, key=hdf_key)
    if target_col not in raw.columns:
        raise KeyError(f"Missing target col {target_col!r} in HDF data.")
    return raw[target_col]
