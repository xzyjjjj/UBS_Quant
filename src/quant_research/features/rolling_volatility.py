from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


def add_rolling_volatility(
    panel: "pd.DataFrame",
    *,
    price_col_candidates: Sequence[str] = ("equity_adj_close", "equity_close"),
    windows: Iterable[int] = (21, 63, 252),
    annualize: bool = True,
    trading_days: int = 252,
    use_log_returns: bool = True,
    ddof: int = 1,
    col_template: str = "equity_rvol_{window}",
) -> "pd.DataFrame":
    """Add rolling realized volatility columns to a market panel.

    Notes:
      - Uses price from the first available column in price_col_candidates.
      - Returns are computed as log returns by default; set use_log_returns=False
        to use simple returns.
      - Volatility is annualized by sqrt(trading_days) when annualize=True.
      - Rolling windows use min_periods=window (i.e., leading values are NaN).
    """
    import numpy as np
    import pandas as pd

    if panel is None or panel.empty:
        return panel

    price_col = next((c for c in price_col_candidates if c in panel.columns), None)
    if not price_col:
        return panel

    out = panel.copy()
    price = pd.to_numeric(out[price_col], errors="coerce")

    if use_log_returns:
        price = price.where(price > 0)
        returns = np.log(price).diff()
    else:
        returns = price.pct_change()

    ann = float(np.sqrt(float(trading_days))) if annualize else 1.0

    for w in windows:
        window = int(w)
        if window <= 1:
            continue
        col = col_template.format(window=window)
        out[col] = returns.rolling(window=window, min_periods=window).std(ddof=ddof) * ann

    return out

