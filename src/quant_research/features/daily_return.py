from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


def add_daily_return(
    panel: "pd.DataFrame",
    *,
    price_col_candidates: Sequence[str] = ("equity_adj_close", "equity_close"),
    out_col: str = "equity_ret_1d",
) -> "pd.DataFrame":
    """Add 1-day simple return: price[t] / price[t-1] - 1."""
    import pandas as pd

    if panel is None or panel.empty:
        return panel

    price_col = next((c for c in price_col_candidates if c in panel.columns), None)
    if not price_col:
        return panel

    out = panel.copy()
    price = pd.to_numeric(out[price_col], errors="coerce")
    out[out_col] = price.pct_change()
    return out

