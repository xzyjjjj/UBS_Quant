from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


def get_trading_calendar(
    raw: dict[str, "pd.DataFrame"], *, calendar_key: str = "equity"
) -> "pd.DatetimeIndex":
    """Get a trading calendar from one of the raw datasets (default: equity).

    Current policy (dependency-free):
      - Use the date index returned by the chosen raw dataset as the "trading calendar".

    Notes:
      - This is pragmatic for pipelines that fetch data via Yahoo/FRED without introducing
        exchange-calendar dependencies.
      - If you need an exchange-accurate calendar (NYSE, etc.), consider wiring in an
        optional provider (e.g. pandas_market_calendars) later.
    """
    import pandas as pd

    df = raw.get(calendar_key)
    if df is None or df.empty:
        return pd.DatetimeIndex([], name="date")

    idx = pd.to_datetime(df.index).normalize()
    idx = idx[~idx.duplicated(keep="last")].sort_values()
    idx.name = "date"
    return idx


def filter_raw_to_calendar(
    raw: dict[str, "pd.DataFrame"], calendar: "pd.DatetimeIndex"
) -> dict[str, "pd.DataFrame"]:
    """Filter each raw frame to calendar dates (drop non-trading days)."""
    import pandas as pd

    if calendar is None or len(calendar) == 0:
        return raw

    cal = pd.to_datetime(calendar).normalize()
    out: dict[str, pd.DataFrame] = {}
    for key, df in raw.items():
        if df is None or df.empty:
            out[key] = df
            continue
        tmp = df.copy()
        tmp.index = pd.to_datetime(tmp.index).normalize()
        # Keep duplicate dates (e.g. option contract-level snapshots); only filter by date membership.
        tmp = tmp.loc[tmp.index.isin(cal)].sort_index()
        tmp.index.name = "date"
        out[key] = tmp
    return out
