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
class PanelConfig:
    start: date | str = "2018-01-01"
    end: date | str = "2024-12-31"
    equity_ticker: str = "JPM"
    vol_ticker: str = "^VIX"
    vol_source: str = "fred"  # "fred" | "yahoo"
    vol_fred_series_id: str = "VIXCLS"
    treasury_series: tuple[str, ...] = DEFAULT_TREASURY_SERIES
    fred_api_key: Optional[str] = None
    alpha_vantage_api_key: Optional[str] = None
    prefer_alpha_vantage_for_equity: bool = False


def fetch_raw(cfg: PanelConfig) -> dict[str, "pd.DataFrame"]:
    """Fetch raw datasets for a simple market panel.

    Returns:
      dict with keys: "equity", "vol", "treasury"
    """
    import pandas as pd

    equity = pd.DataFrame()
    vol = pd.DataFrame()
    treasury = pd.DataFrame()

    # Equity: Yahoo first (default). If rate-limited, fall back to Alpha Vantage if configured.
    try:
        if cfg.prefer_alpha_vantage_for_equity:
            if not cfg.alpha_vantage_api_key:
                raise ValueError("prefer_alpha_vantage_for_equity=True requires alpha_vantage_api_key.")
            equity = fetch_alpha_vantage_daily_adjusted(
                cfg.equity_ticker, cfg.start, cfg.end, api_key=cfg.alpha_vantage_api_key
            )
        else:
            equity = fetch_yahoo_daily(cfg.equity_ticker, cfg.start, cfg.end, auto_adjust=False)
    except Exception:
        if cfg.alpha_vantage_api_key:
            equity = fetch_alpha_vantage_daily_adjusted(
                cfg.equity_ticker, cfg.start, cfg.end, api_key=cfg.alpha_vantage_api_key
            )

    # Volatility index: prefer FRED (VIXCLS) to avoid Yahoo rate limits.
    try:
        if cfg.vol_source == "fred":
            vol = fetch_fred_series_many(
                (cfg.vol_fred_series_id,),
                cfg.start,
                cfg.end,
                api_key=cfg.fred_api_key,
            )
        elif cfg.vol_source == "yahoo":
            vol = fetch_yahoo_daily(cfg.vol_ticker, cfg.start, cfg.end, auto_adjust=False)
        else:
            raise ValueError(f"Unknown vol_source: {cfg.vol_source!r}")
    except Exception:
        vol = pd.DataFrame()

    # Treasury yields: FRED.
    try:
        treasury = fetch_fred_series_many(
            cfg.treasury_series, cfg.start, cfg.end, api_key=cfg.fred_api_key
        )
    except Exception:
        treasury = pd.DataFrame()

    return {"equity": equity, "vol": vol, "treasury": treasury}


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


def build_panel(raw: dict[str, "pd.DataFrame"]) -> "pd.DataFrame":
    """Build a single date-indexed panel (outer-joined)."""
    import pandas as pd

    equity = _normalize_price_frame(raw.get("equity", pd.DataFrame()), "equity")
    vol_raw = raw.get("vol", pd.DataFrame())
    if vol_raw is not None and not vol_raw.empty and list(vol_raw.columns) == ["VIXCLS"]:
        vol = vol_raw.rename(columns={"VIXCLS": "vol_close"})
        vol.index = pd.to_datetime(vol.index)
        vol.index.name = "date"
    else:
        vol = _normalize_price_frame(vol_raw, "vol")
    treasury = _normalize_treasury_frame(raw.get("treasury", pd.DataFrame()))

    frames = [x for x in (equity, vol, treasury) if not x.empty]
    if not frames:
        out = pd.DataFrame()
        out.index.name = "date"
        return out

    panel = pd.concat(frames, axis=1, join="outer").sort_index()
    panel.index.name = "date"
    return panel


def save_outputs(
    output_dir: Path,
    raw: dict[str, "pd.DataFrame"],
    panel: "pd.DataFrame",
) -> None:
    """Save raw and processed outputs under output_dir (raw/ + processed/)."""
    import pandas as pd

    raw_dir = output_dir / "raw"
    processed_dir = output_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw.get("equity", pd.DataFrame()).to_csv(raw_dir / "equity_daily.csv", index=True)
    raw.get("vol", pd.DataFrame()).to_csv(raw_dir / "vol_daily.csv", index=True)
    raw.get("treasury", pd.DataFrame()).to_csv(raw_dir / "treasury_yields.csv", index=True)

    if panel is not None and not panel.empty:
        panel.to_csv(processed_dir / "panel.csv", index=True)
        try:
            panel.to_parquet(processed_dir / "panel.parquet", index=True)
        except Exception:
            # Parquet is optional (pyarrow/fastparquet may be missing).
            pass
