from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


def add_dividend_growth(
    panel: "pd.DataFrame",
    equity_raw: "pd.DataFrame | None",
    *,
    dividend_col_candidates: Sequence[str] = ("Dividends", "Dividend"),
    out_col: str = "equity_div_growth",
    fill_method: str = "ffill",
) -> "pd.DataFrame":
    """Add dividend growth based on consecutive dividend events (A).

    Definition (event-time):
      growth[t_event] = dividend[t_event] / dividend[prev_event] - 1

    For non-dividend days:
      - default fill_method="ffill": forward-fill the most recent computed growth value
      - fill_method="none": keep NaN except on dividend event days
    """
    import numpy as np
    import pandas as pd

    if panel is None or panel.empty:
        return panel
    if equity_raw is None or equity_raw.empty:
        return panel

    div_col = next((c for c in dividend_col_candidates if c in equity_raw.columns), None)
    if not div_col:
        return panel

    equity = equity_raw.copy()
    equity.index = pd.to_datetime(equity.index)
    equity = equity[~equity.index.duplicated(keep="last")].sort_index()

    div = pd.to_numeric(equity[div_col], errors="coerce").fillna(0.0)
    event_div = div[div > 0]
    if event_div.empty:
        return panel

    prev = event_div.shift(1)
    growth_event = (event_div / prev) - 1.0
    growth_event = growth_event.where(prev > 0)

    growth_daily = growth_event.reindex(pd.to_datetime(panel.index))
    if fill_method == "ffill":
        growth_daily = growth_daily.ffill()
    elif fill_method == "none":
        pass
    else:
        raise ValueError(f"Unknown fill_method: {fill_method!r}. Use 'ffill' or 'none'.")

    out = panel.copy()
    out[out_col] = growth_daily.astype(float)
    out[out_col] = out[out_col].replace([np.inf, -np.inf], np.nan)
    return out

