"""Portfolio allocation helpers."""

from __future__ import annotations

import pandas as pd


def equity_curve(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0.0)).cumprod()
