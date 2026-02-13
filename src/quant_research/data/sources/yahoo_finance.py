from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


def fetch_yahoo_daily(
    ticker: str,
    start: date | str,
    end: date | str,
    *,
    auto_adjust: bool = False,
) -> "pd.DataFrame":
    """Fetch Yahoo Finance daily OHLCV via yfinance.

    Notes:
    - Uses `yfinance`, which is a community wrapper (not an official Yahoo API).
    - Returns a DataFrame indexed by date with columns like Open/High/Low/Close/Adj Close/Volume.
    """
    try:
        import yfinance as yf  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: yfinance. Install it with `pip install yfinance`."
        ) from e

    import pandas as pd

    df = yf.download(
        tickers=ticker,
        start=str(start),
        end=str(end),
        interval="1d",
        auto_adjust=auto_adjust,
        progress=False,
        actions=False,
        threads=True,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance can return MultiIndex when tickers is list-like; normalize.
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df.sort_index()
