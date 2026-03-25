from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


def add_news_sentiment(
    panel: "pd.DataFrame",
    *,
    sentiment_csv_path: Path,
    score_col: str = "news_sent_01",
    out_col: str = "news_sent_01",
    out_ewm_col: str = "news_sent_01_ewm7",
    ewm_span: int = 7,
    fill_method: str = "ffill",
) -> "pd.DataFrame":
    """Join daily news sentiment (0-1) into panel.

    Expects sentiment_csv_path to contain a `date` column and score_col column.
    """
    import numpy as np
    import pandas as pd

    if panel is None or panel.empty:
        return panel
    if not sentiment_csv_path.exists():
        return panel

    df = pd.read_csv(sentiment_csv_path, parse_dates=["date"])
    if df.empty or "date" not in df.columns or score_col not in df.columns:
        return panel

    s = pd.to_numeric(df[score_col], errors="coerce")
    tmp = pd.DataFrame({"date": pd.to_datetime(df["date"]), out_col: s})
    tmp = tmp.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last").sort_values("date")
    tmp = tmp.set_index("date")

    out = panel.copy()
    joined = out.join(tmp[[out_col]], how="left")
    if fill_method == "ffill":
        joined[out_col] = joined[out_col].ffill()
    elif fill_method == "none":
        pass
    else:
        raise ValueError(f"Unknown fill_method: {fill_method!r}. Use 'ffill' or 'none'.")

    span = int(ewm_span)
    if span > 1:
        joined[out_ewm_col] = joined[out_col].ewm(span=span, adjust=False).mean()

    joined[out_col] = joined[out_col].replace([np.inf, -np.inf], np.nan)
    if out_ewm_col in joined.columns:
        joined[out_ewm_col] = joined[out_ewm_col].replace([np.inf, -np.inf], np.nan)
    return joined

