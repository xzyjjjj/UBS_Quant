from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


def add_equity_vol_correlation(
    panel: "pd.DataFrame",
    *,
    equity_ret_col_candidates: Sequence[str] = ("equity_ret_1d",),
    vol_price_col_candidates: Sequence[str] = ("vol_close",),
    windows: Iterable[int] = (21, 63, 252),
    out_col_template: str = "equity_vix_corr_{window}",
) -> "pd.DataFrame":
    """Add rolling correlation between equity returns and VIX returns.

    Uses:
      - equity returns: equity_ret_col_candidates (expects daily simple return)
      - VIX returns: pct_change(vol_close)
    """
    import numpy as np
    import pandas as pd

    if panel is None or panel.empty:
        return panel

    equity_ret_col = next((c for c in equity_ret_col_candidates if c in panel.columns), None)
    vol_price_col = next((c for c in vol_price_col_candidates if c in panel.columns), None)
    if not equity_ret_col or not vol_price_col:
        return panel

    out = panel.copy()

    equity_ret = pd.to_numeric(out[equity_ret_col], errors="coerce")
    vol_price = pd.to_numeric(out[vol_price_col], errors="coerce")
    vol_ret = vol_price.pct_change()

    for w in windows:
        window = int(w)
        if window <= 1:
            continue
        col = out_col_template.format(window=window)
        corr = equity_ret.rolling(window=window, min_periods=window).corr(vol_ret)
        out[col] = corr.replace([np.inf, -np.inf], np.nan)

    return out

