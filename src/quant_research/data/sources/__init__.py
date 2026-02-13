"""External data sources (Week1 and beyond)."""

from __future__ import annotations

from .fred import DEFAULT_TREASURY_SERIES, fetch_fred_series, fetch_fred_series_many
from .yahoo_finance import fetch_yahoo_daily
from .alpha_vantage import fetch_alpha_vantage_daily_adjusted

__all__ = [
    "DEFAULT_TREASURY_SERIES",
    "fetch_alpha_vantage_daily_adjusted",
    "fetch_fred_series",
    "fetch_fred_series_many",
    "fetch_yahoo_daily",
]

