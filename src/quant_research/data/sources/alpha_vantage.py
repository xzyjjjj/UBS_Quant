from __future__ import annotations

from datetime import date
from typing import Optional

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore


def _require_requests() -> None:
    if requests is None:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: requests. Install it with `pip install requests`."
        )


def fetch_alpha_vantage_daily_adjusted(
    symbol: str,
    start: date | str,
    end: date | str,
    *,
    api_key: str,
    outputsize: str = "full",
) -> "pd.DataFrame":
    """Fetch Alpha Vantage TIME_SERIES_DAILY_ADJUSTED for equities.

    Returns a DataFrame indexed by date with normalized columns:
    Open, High, Low, Close, Adj Close, Volume, Dividend, Split Coefficient.
    """
    _require_requests()
    if not api_key:
        raise ValueError("alpha_vantage api_key is required.")

    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": symbol,
        "outputsize": outputsize,
        "apikey": api_key,
    }
    resp = requests.get("https://www.alphavantage.co/query", params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    ts = payload.get("Time Series (Daily)")
    if not isinstance(ts, dict):
        msg = payload.get("Note") or payload.get("Error Message") or str(payload)[:200]
        raise RuntimeError(f"Alpha Vantage response missing daily series: {msg}")

    rows = []
    for dt, vals in ts.items():
        rows.append(
            {
                "date": dt,
                "Open": float(vals.get("1. open", "nan")),
                "High": float(vals.get("2. high", "nan")),
                "Low": float(vals.get("3. low", "nan")),
                "Close": float(vals.get("4. close", "nan")),
                "Adj Close": float(vals.get("5. adjusted close", "nan")),
                "Volume": float(vals.get("6. volume", "nan")),
                "Dividend": float(vals.get("7. dividend amount", "nan")),
                "Split Coefficient": float(vals.get("8. split coefficient", "nan")),
            }
        )

    import pandas as pd

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    start_ts = pd.to_datetime(str(start))
    end_ts = pd.to_datetime(str(end))
    df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
    df.index.name = "date"
    return df
