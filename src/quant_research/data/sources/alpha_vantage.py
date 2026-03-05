from __future__ import annotations

from datetime import date
import time
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


def fetch_alpha_vantage_historical_options(
    symbol: str,
    start: date | str,
    end: date | str,
    *,
    api_key: str,
    min_interval_secs: float = 0.85,
    max_retries: int = 4,
    timeout: int = 30,
    progress_every: int = 50,
) -> "pd.DataFrame":
    """Fetch Alpha Vantage HISTORICAL_OPTIONS daily snapshots in a date range.

    Notes:
      - Calls one request per business day with `date=YYYY-MM-DD`.
      - Returns a contract-level DataFrame indexed by `date`.
      - Empty result is returned when no options payload is available.
    """
    _require_requests()
    if not api_key:
        raise ValueError("alpha_vantage api_key is required.")

    import pandas as pd

    start_ts = pd.to_datetime(str(start)).normalize()
    end_ts = pd.to_datetime(str(end)).normalize()
    if end_ts < start_ts:
        raise ValueError("end must be >= start")

    dates = pd.bdate_range(start=start_ts, end=end_ts)
    if len(dates) == 0:
        out = pd.DataFrame()
        out.index.name = "date"
        return out

    rows: list[dict[str, object]] = []
    last_request_ts: float | None = None
    base_url = "https://www.alphavantage.co/query"

    total_days = int(len(dates))
    for i, d in enumerate(dates, start=1):
        query_day = d.strftime("%Y-%m-%d")
        attempt = 0
        got_data_for_day = False
        while True:
            if last_request_ts is not None and min_interval_secs > 0:
                elapsed = time.time() - last_request_ts
                if elapsed < min_interval_secs:
                    time.sleep(min_interval_secs - elapsed)

            params = {
                "function": "HISTORICAL_OPTIONS",
                "symbol": symbol,
                "date": query_day,
                "apikey": api_key,
            }
            try:
                last_request_ts = time.time()
                resp = requests.get(base_url, params=params, timeout=int(timeout))
                resp.raise_for_status()
                payload = resp.json()
            except Exception as e:
                attempt += 1
                if attempt <= int(max_retries):
                    time.sleep(min(60.0, (1.8 ** attempt)))
                    continue
                print(f"[WARN] options request failed after retries: date={query_day} err={type(e).__name__}: {e}")
                break

            data = payload.get("data")
            if isinstance(data, list):
                for rec in data:
                    if not isinstance(rec, dict):
                        continue
                    row = dict(rec)
                    row.setdefault("symbol", symbol)
                    row.setdefault("requested_date", query_day)
                    rows.append(row)
                got_data_for_day = True
                break

            note = str(payload.get("Note") or payload.get("Information") or "").strip()
            err_msg = str(payload.get("Error Message") or "").strip()
            rate_limited = "rate limit" in note.lower() or "frequency" in note.lower()
            if rate_limited and attempt < int(max_retries):
                attempt += 1
                time.sleep(min(60.0, (1.8 ** attempt)))
                continue
            if note:
                # No data note (or non-rate-limit note), continue next day.
                break
            if err_msg:
                print(f"[WARN] options API error: date={query_day} message={err_msg}")
            break

        if int(progress_every) > 0 and (i % int(progress_every) == 0 or i == total_days):
            status = "ok" if got_data_for_day else "empty_or_failed"
            print(f"[INFO] options progress: {i}/{total_days} ({(100.0 * i / total_days):.1f}%) day_status={status}")

    df = pd.DataFrame(rows)
    if df.empty:
        df.index.name = "date"
        return df

    if "date" in df.columns:
        dt = pd.to_datetime(df["date"], errors="coerce")
    else:
        dt = pd.to_datetime(df.get("requested_date"), errors="coerce")
    df["date"] = dt.dt.normalize()
    df = df.dropna(subset=["date"]).copy()

    # Best-effort numeric parsing for options fields.
    for col in (
        "strike",
        "last",
        "mark",
        "bid",
        "ask",
        "bid_size",
        "ask_size",
        "volume",
        "open_interest",
        "implied_volatility",
        "delta",
        "gamma",
        "theta",
        "vega",
        "rho",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "expiration" in df.columns:
        df["expiration"] = pd.to_datetime(df["expiration"], errors="coerce").dt.normalize()
    if "type" in df.columns:
        df["type"] = df["type"].astype(str).str.lower()

    df = df.sort_values(["date"])
    df = df.set_index("date")
    return df
