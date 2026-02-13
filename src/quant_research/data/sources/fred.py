from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import time
from typing import Iterable, Optional

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore


# Common U.S. Treasury "constant maturity" yield series (daily, percent).
# Reference: FRED series IDs prefixed with DGS*.
DEFAULT_TREASURY_SERIES: tuple[str, ...] = (
    "DGS1MO",
    "DGS3MO",
    "DGS6MO",
    "DGS1",
    "DGS2",
    "DGS3",
    "DGS5",
    "DGS7",
    "DGS10",
    "DGS20",
    "DGS30",
)


@dataclass(frozen=True)
class FREDConfig:
    api_key: Optional[str] = None


def _require_requests() -> None:
    if requests is None:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: requests. Install it with `pip install requests`."
        )


def _get_json(url: str, params: dict, *, retries: int = 3, timeout: int = 30) -> dict:
    _require_requests()
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(8, 2**attempt))
                continue
            if resp.status_code == 400:
                hint = ""
                if "api_key" not in params:
                    hint = " (hint: set env FRED_API_KEY or pass --fred-api-key)"
                raise RuntimeError(f"FRED 400 Bad Request{hint}: {resp.text[:200]}")
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_exc = e
            time.sleep(min(8, 2**attempt))
            continue
    raise RuntimeError(f"FRED request failed after {retries} attempts: {last_exc}")


def fetch_fred_series(
    series_id: str,
    start: date | str,
    end: date | str,
    *,
    api_key: Optional[str] = None,
) -> "pd.DataFrame":
    """Fetch a single FRED series as a DataFrame indexed by date.

    Returns a DataFrame with one column named `series_id` and a DatetimeIndex.
    Missing values are returned as NaN.
    """
    _require_requests()
    import pandas as pd

    start_s = str(start)
    end_s = str(end)

    params = {
        "series_id": series_id,
        "file_type": "json",
        "observation_start": start_s,
        "observation_end": end_s,
    }
    if api_key:
        params["api_key"] = api_key

    payload = _get_json(
        "https://api.stlouisfed.org/fred/series/observations",
        params,
        retries=3,
        timeout=30,
    )

    observations = payload.get("observations", [])
    rows = []
    for obs in observations:
        dt = obs.get("date")
        val = obs.get("value")
        if dt is None:
            continue
        if val in (None, "."):
            v = float("nan")
        else:
            try:
                v = float(val)
            except Exception:
                v = float("nan")
        rows.append((dt, v))

    df = pd.DataFrame(rows, columns=["date", series_id])
    if df.empty:
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df


def fetch_fred_series_many(
    series_ids: Iterable[str],
    start: date | str,
    end: date | str,
    *,
    api_key: Optional[str] = None,
) -> "pd.DataFrame":
    """Fetch many FRED series and outer-join them on date."""
    import pandas as pd

    frames: list[pd.DataFrame] = []
    for sid in series_ids:
        frames.append(fetch_fred_series(sid, start, end, api_key=api_key))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1, join="outer").sort_index()
    out.index.name = "date"
    return out
