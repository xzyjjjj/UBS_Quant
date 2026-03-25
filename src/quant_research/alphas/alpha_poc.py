from __future__ import annotations

import pandas as pd

from quant_research.alphas.registry import register_alpha
from quant_research.features.support_resistance import poc_bias_long


@register_alpha(
    "poc_bias_long",
    freq="minute",
    lookahead=False,
    inputs=("close", "volume"),
    description="Long bias from POC below/above computed on minute data.",
)
def alpha_poc_bias_long(readouts: pd.DataFrame) -> pd.Series:
    return poc_bias_long(
        readouts,
        window=240,
        bins=40,
        price_col=None,
        volume_col="volume",
        shift=1,
        groupby_level=None,
    )
