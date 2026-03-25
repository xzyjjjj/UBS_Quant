from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


def add_rate_momentum(
    panel: "pd.DataFrame",
    *,
    rate_col_candidates: Sequence[str] = ("DGS10",),
    horizons: Iterable[int] = (21, 63),
    zscore_window: int = 252,
    out_col_template: str = "rate_mom_{rate}_{h}",
    out_z_col_template: str = "rate_mom_{rate}_{h}_z{z}",
) -> "pd.DataFrame":
    """Add rate momentum features using diffs, plus rolling z-scores.

    - Momentum (horizon h): rate[t] - rate[t-h]
    - Z-score: (mom - mean(mom, zscore_window)) / std(mom, zscore_window)
    """
    import numpy as np
    import pandas as pd

    if panel is None or panel.empty:
        return panel

    rate_col = next((c for c in rate_col_candidates if c in panel.columns), None)
    if not rate_col:
        return panel

    out = panel.copy()
    rate = pd.to_numeric(out[rate_col], errors="coerce")

    z = int(zscore_window)
    if z <= 1:
        raise ValueError("zscore_window must be > 1")

    for h_raw in horizons:
        h = int(h_raw)
        if h <= 0:
            continue
        mom = rate.diff(h)
        out_col = out_col_template.format(rate=rate_col.lower(), h=h)
        out[out_col] = mom.replace([np.inf, -np.inf], np.nan)

        mean = mom.rolling(window=z, min_periods=z).mean()
        std = mom.rolling(window=z, min_periods=z).std(ddof=0)
        zcol = out_z_col_template.format(rate=rate_col.lower(), h=h, z=z)
        out[zcol] = ((mom - mean) / std).replace([np.inf, -np.inf], np.nan)

    return out

