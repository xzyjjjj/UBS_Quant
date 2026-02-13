from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

from quant_research.data.sources import (
    DEFAULT_TREASURY_SERIES,
    fetch_alpha_vantage_daily_adjusted,
    fetch_fred_series_many,
    fetch_yahoo_daily,
)


@dataclass(frozen=True)
class Week1Config:
    start: date | str = "2018-01-01"
    end: date | str = "2024-12-31"
    jpm_ticker: str = "JPM"
    vix_ticker: str = "^VIX"
    vix_source: str = "fred"  # "fred" | "yahoo"
    vix_fred_series_id: str = "VIXCLS"
    treasury_series: tuple[str, ...] = DEFAULT_TREASURY_SERIES
    fred_api_key: Optional[str] = None
    alpha_vantage_api_key: Optional[str] = None
    prefer_alpha_vantage_for_jpm: bool = False


def fetch_week1_raw(cfg: Week1Config) -> dict[str, "pd.DataFrame"]:
    """Fetch Week1 raw datasets.

    Returns:
      dict with keys: "jpm", "vix", "treasury"
    """
    import pandas as pd

    jpm = pd.DataFrame()
    vix = pd.DataFrame()
    treasury = pd.DataFrame()

    # JPM: Yahoo first (default). If rate-limited, fall back to Alpha Vantage if configured.
    try:
        if cfg.prefer_alpha_vantage_for_jpm:
            if not cfg.alpha_vantage_api_key:
                raise ValueError("prefer_alpha_vantage_for_jpm=True requires alpha_vantage_api_key.")
            jpm = fetch_alpha_vantage_daily_adjusted(
                cfg.jpm_ticker, cfg.start, cfg.end, api_key=cfg.alpha_vantage_api_key
            )
        else:
            jpm = fetch_yahoo_daily(cfg.jpm_ticker, cfg.start, cfg.end, auto_adjust=False)
    except Exception:
        if cfg.alpha_vantage_api_key:
            jpm = fetch_alpha_vantage_daily_adjusted(
                cfg.jpm_ticker, cfg.start, cfg.end, api_key=cfg.alpha_vantage_api_key
            )

    # VIX: prefer FRED (VIXCLS) to avoid Yahoo rate limits.
    try:
        if cfg.vix_source == "fred":
            vix = fetch_fred_series_many(
                (cfg.vix_fred_series_id,),
                cfg.start,
                cfg.end,
                api_key=cfg.fred_api_key,
            )
        elif cfg.vix_source == "yahoo":
            vix = fetch_yahoo_daily(cfg.vix_ticker, cfg.start, cfg.end, auto_adjust=False)
        else:
            raise ValueError(f"Unknown vix_source: {cfg.vix_source!r}")
    except Exception:
        vix = pd.DataFrame()

    # Treasury yields: FRED.
    try:
        treasury = fetch_fred_series_many(
            cfg.treasury_series, cfg.start, cfg.end, api_key=cfg.fred_api_key
        )
    except Exception:
        treasury = pd.DataFrame()

    return {"jpm": jpm, "vix": vix, "treasury": treasury}


def _normalize_price_frame(df: "pd.DataFrame", prefix: str) -> "pd.DataFrame":
    import pandas as pd

    if df is None or df.empty:
        out = pd.DataFrame()
        out.index.name = "date"
        return out

    tmp = df.copy()
    tmp.index = pd.to_datetime(tmp.index)
    tmp = tmp[~tmp.index.duplicated(keep="last")].sort_index()
    tmp.index.name = "date"

    mapping = {
        "Open": f"{prefix}_open",
        "High": f"{prefix}_high",
        "Low": f"{prefix}_low",
        "Close": f"{prefix}_close",
        "Adj Close": f"{prefix}_adj_close",
        "Volume": f"{prefix}_volume",
    }
    keep_cols = [c for c in mapping.keys() if c in tmp.columns]
    tmp = tmp[keep_cols].rename(columns={c: mapping[c] for c in keep_cols})
    return tmp


def _normalize_treasury_frame(df: "pd.DataFrame") -> "pd.DataFrame":
    import pandas as pd

    if df is None or df.empty:
        out = pd.DataFrame()
        out.index.name = "date"
        return out
    tmp = df.copy()
    tmp.index = pd.to_datetime(tmp.index)
    tmp = tmp[~tmp.index.duplicated(keep="last")].sort_index()
    tmp.index.name = "date"
    # Keep as percent level (e.g., 4.12 means 4.12%).
    return tmp


def build_week1_panel(raw: dict[str, "pd.DataFrame"]) -> "pd.DataFrame":
    import pandas as pd

    """Build a single date-indexed panel for Week1 (outer-joined)."""
    jpm = _normalize_price_frame(raw.get("jpm", pd.DataFrame()), "jpm")
    vix_raw = raw.get("vix", pd.DataFrame())
    if vix_raw is not None and not vix_raw.empty and list(vix_raw.columns) == ["VIXCLS"]:
        vix = vix_raw.rename(columns={"VIXCLS": "vix_close"})
        vix.index = pd.to_datetime(vix.index)
        vix.index.name = "date"
    else:
        vix = _normalize_price_frame(vix_raw, "vix")
    treasury = _normalize_treasury_frame(raw.get("treasury", pd.DataFrame()))

    frames = [x for x in (jpm, vix, treasury) if not x.empty]
    if not frames:
        out = pd.DataFrame()
        out.index.name = "date"
        return out

    panel = pd.concat(frames, axis=1, join="outer").sort_index()
    panel.index.name = "date"
    return panel


def save_week1_outputs(
    output_dir: Path,
    raw: dict[str, "pd.DataFrame"],
    panel: "pd.DataFrame",
) -> None:
    """Save raw and processed outputs under output_dir/week1/."""
    import pandas as pd

    raw_dir = output_dir / "raw"
    processed_dir = output_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw.get("jpm", pd.DataFrame()).to_csv(raw_dir / "jpm_daily.csv", index=True)
    raw.get("vix", pd.DataFrame()).to_csv(raw_dir / "vix_daily.csv", index=True)
    raw.get("treasury", pd.DataFrame()).to_csv(raw_dir / "treasury_yields.csv", index=True)

    if panel is not None and not panel.empty:
        panel.to_csv(processed_dir / "panel.csv", index=True)
        try:
            panel.to_parquet(processed_dir / "panel.parquet", index=True)
        except Exception:
            # Parquet is optional (pyarrow/fastparquet may be missing).
            pass
