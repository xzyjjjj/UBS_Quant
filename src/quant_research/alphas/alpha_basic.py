from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research.alphas.registry import register_alpha


@register_alpha(
    "default_st",
    freq="day",
    lookahead=False,
    inputs=("T_last_close", "T_vwap"),
    description="Sign of last_close / vwap > 1 (contrarian alpha).",
)
def alpha_default_st(readouts: pd.DataFrame) -> pd.Series:
    ratio = np.array(readouts["T_last_close"]) / np.array(readouts["T_vwap"])
    return pd.Series(np.where(ratio > 1.0, -1, 1), index=readouts.index)
