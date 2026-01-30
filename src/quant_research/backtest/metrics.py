"""Backtest metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, annualization: float = 250.0) -> float:
    arr = returns.dropna().to_numpy()
    if arr.size == 0:
        return float("nan")
    std = np.std(arr)
    if std == 0:
        return float("nan")
    return float(np.sqrt(annualization) * np.mean(arr) / std)
