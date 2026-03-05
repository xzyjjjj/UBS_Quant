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
    fetch_alpha_vantage_historical_options,
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
    include_options: bool = True
    options_symbol: Optional[str] = None
    options_min_interval_secs: float = 0.85
    options_timeout_secs: int = 60
    options_max_retries: int = 6
    options_progress_every: int = 50


def fetch_raw(cfg: PanelConfig) -> dict[str, "pd.DataFrame"]:
    """Fetch raw datasets for a simple market panel.

    Returns:
      dict with keys: "equity", "vol", "treasury", "options"
    """
    import pandas as pd

    equity = pd.DataFrame()
    vol = pd.DataFrame()
    treasury = pd.DataFrame()
    options = pd.DataFrame()

    # Equity: Yahoo first (default). If rate-limited, fall back to Alpha Vantage if configured.
    try:
        if cfg.prefer_alpha_vantage_for_equity:
            if not cfg.alpha_vantage_api_key:
                raise ValueError("prefer_alpha_vantage_for_equity=True requires alpha_vantage_api_key.")
            equity = fetch_alpha_vantage_daily_adjusted(
                cfg.equity_ticker, cfg.start, cfg.end, api_key=cfg.alpha_vantage_api_key
            )
        else:
            equity = fetch_yahoo_daily(
                cfg.equity_ticker,
                cfg.start,
                cfg.end,
                auto_adjust=False,
                actions=True,
            )
            if equity is None or equity.empty:
                raise RuntimeError("Yahoo returned empty dataframe (likely rate-limited or symbol invalid).")
    except Exception as e:
        print(
            f"[WARN] Fetch equity failed (source=yahoo, ticker={cfg.equity_ticker}): {type(e).__name__}: {e}"
        )
        if cfg.alpha_vantage_api_key:
            try:
                equity = fetch_alpha_vantage_daily_adjusted(
                    cfg.equity_ticker, cfg.start, cfg.end, api_key=cfg.alpha_vantage_api_key
                )
            except Exception as e2:
                print(
                    f"[WARN] Fetch equity failed (source=alpha_vantage, symbol={cfg.equity_ticker}): {type(e2).__name__}: {e2}"
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
            if vol is None or vol.empty:
                raise RuntimeError("Yahoo returned empty dataframe (likely rate-limited or symbol invalid).")
        else:
            raise ValueError(f"Unknown vol_source: {cfg.vol_source!r}")
    except Exception as e:
        if cfg.vol_source == "yahoo":
            print(
                f"[WARN] Fetch vol failed (source=yahoo, ticker={cfg.vol_ticker}): {type(e).__name__}: {e}"
            )
        else:
            print(
                f"[WARN] Fetch vol failed (source=fred, series_id={cfg.vol_fred_series_id}): {type(e).__name__}: {e}"
            )
        vol = pd.DataFrame()

    # Treasury yields: FRED.
    try:
        treasury = fetch_fred_series_many(
            cfg.treasury_series, cfg.start, cfg.end, api_key=cfg.fred_api_key
        )
    except Exception as e:
        print(
            f"[WARN] Fetch treasury failed (source=fred, series_ids={','.join(cfg.treasury_series)}): {type(e).__name__}: {e}"
        )
        treasury = pd.DataFrame()

    # US equity options history from Alpha Vantage.
    if cfg.include_options:
        opt_symbol = str(cfg.options_symbol or cfg.equity_ticker).strip().upper()
        if not cfg.alpha_vantage_api_key:
            print("[WARN] Skip options fetch: alpha_vantage_api_key is missing.")
        else:
            try:
                print(
                    f"[INFO] Fetch options history (source=alpha_vantage, symbol={opt_symbol}, "
                    f"start={cfg.start}, end={cfg.end})"
                )
                options = fetch_alpha_vantage_historical_options(
                    opt_symbol,
                    cfg.start,
                    cfg.end,
                    api_key=cfg.alpha_vantage_api_key,
                    min_interval_secs=float(cfg.options_min_interval_secs),
                    timeout=int(cfg.options_timeout_secs),
                    max_retries=int(cfg.options_max_retries),
                    progress_every=int(cfg.options_progress_every),
                )
            except Exception as e:
                print(
                    f"[WARN] Fetch options failed (source=alpha_vantage, symbol={opt_symbol}): "
                    f"{type(e).__name__}: {e}"
                )
                options = pd.DataFrame()

    return {"equity": equity, "vol": vol, "treasury": treasury, "options": options}


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


def _aggregate_options_daily(df: "pd.DataFrame") -> "pd.DataFrame":
    import pandas as pd

    if df is None or df.empty:
        out = pd.DataFrame()
        out.index.name = "date"
        return out

    tmp = df.copy()
    if "date" in tmp.columns:
        tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce").dt.normalize()
        tmp = tmp.dropna(subset=["date"]).set_index("date")
    else:
        tmp.index = pd.to_datetime(tmp.index, errors="coerce")
        tmp = tmp[~tmp.index.isna()]
        tmp.index = tmp.index.normalize()
    tmp.index.name = "date"

    for col in ("volume", "open_interest", "implied_volatility", "delta", "gamma", "theta", "vega", "rho", "mark"):
        if col in tmp.columns:
            tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
    if "type" in tmp.columns:
        tmp["type"] = tmp["type"].astype(str).str.lower()

    g = tmp.groupby(level=0)
    out = pd.DataFrame(index=g.size().index)
    out.index.name = "date"
    out["opt_contracts"] = g.size().astype(float)

    if "volume" in tmp.columns:
        out["opt_volume_sum"] = g["volume"].sum(min_count=1)
    if "open_interest" in tmp.columns:
        out["opt_open_interest_sum"] = g["open_interest"].sum(min_count=1)
    if "implied_volatility" in tmp.columns:
        out["opt_iv_mean"] = g["implied_volatility"].mean()
        out["opt_iv_median"] = g["implied_volatility"].median()
    if "mark" in tmp.columns:
        out["opt_mark_mean"] = g["mark"].mean()

    if "type" in tmp.columns and "implied_volatility" in tmp.columns:
        calls = tmp[tmp["type"] == "call"].groupby(level=0)["implied_volatility"].mean().rename("opt_iv_call_mean")
        puts = tmp[tmp["type"] == "put"].groupby(level=0)["implied_volatility"].mean().rename("opt_iv_put_mean")
        out = out.join(calls, how="left").join(puts, how="left")

    return out.sort_index()


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
    options = _aggregate_options_daily(raw.get("options", pd.DataFrame()))

    frames = [x for x in (equity, vol, treasury, options) if not x.empty]
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
    """Save raw and processed outputs under output_dir (raw/ + processed/).

    Deprecated: prefer save_raw_outputs/save_panel_outputs so fetching, panel
    build, and feature engineering can be run independently.
    """
    save_raw_outputs(output_dir, raw)
    save_panel_outputs(output_dir, panel)


def save_raw_outputs(
    output_dir: Path,
    raw: dict[str, "pd.DataFrame"],
    *,
    overwrite: bool = False,
) -> None:
    """Save raw outputs under output_dir/raw/.

    By default this is non-destructive:
      - If a raw file already exists, it will NOT be overwritten unless overwrite=True.
      - Even with overwrite=True, an empty dataframe will NOT overwrite an existing file.
    """
    import pandas as pd

    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    def _maybe_write(df: "pd.DataFrame", path: Path) -> None:
        if path.exists():
            if df is None or df.empty:
                print(f"[WARN] Skip writing empty raw dataframe to existing file: {path}")
                return
            if not overwrite:
                print(f"[INFO] Raw cache exists; skip overwrite: {path}")
                return
        df = df if df is not None else pd.DataFrame()
        df.to_csv(path, index=True)

    _maybe_write(raw.get("equity", pd.DataFrame()), raw_dir / "equity_daily.csv")
    _maybe_write(raw.get("vol", pd.DataFrame()), raw_dir / "vol_daily.csv")
    _maybe_write(raw.get("treasury", pd.DataFrame()), raw_dir / "treasury_yields.csv")
    _maybe_write(raw.get("options", pd.DataFrame()), raw_dir / "options_history.csv")


def save_panel_outputs(output_dir: Path, panel: "pd.DataFrame") -> None:
    """Save processed panel outputs under output_dir/processed/."""
    processed_dir = output_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    if panel is not None and not panel.empty:
        panel.to_csv(processed_dir / "panel.csv", index=True)
        try:
            panel.to_parquet(processed_dir / "panel.parquet", index=True)
        except Exception:
            # Parquet is optional (pyarrow/fastparquet may be missing).
            pass


def load_raw_outputs(output_dir: Path) -> dict[str, "pd.DataFrame"]:
    """Load raw outputs from output_dir/raw/ (if present)."""
    import pandas as pd

    raw_dir = output_dir / "raw"
    out: dict[str, pd.DataFrame] = {
        "equity": pd.DataFrame(),
        "vol": pd.DataFrame(),
        "treasury": pd.DataFrame(),
        "options": pd.DataFrame(),
    }
    paths = {
        "equity": raw_dir / "equity_daily.csv",
        "vol": raw_dir / "vol_daily.csv",
        "treasury": raw_dir / "treasury_yields.csv",
        "options": raw_dir / "options_history.csv",
    }
    for key, path in paths.items():
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index.name = "date"
        out[key] = df
    return out


def load_panel_output(output_dir: Path) -> "pd.DataFrame":
    """Load processed panel from output_dir/processed/panel.csv."""
    import pandas as pd

    panel_path = output_dir / "processed" / "panel.csv"
    if not panel_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(panel_path, index_col=0, parse_dates=True)
    df.index.name = "date"
    return df
