#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

try:
    from quant_research.data.sources.fred import DEFAULT_TREASURY_SERIES
    from quant_research.data.sources.market_panel import (
        PanelConfig,
        build_panel,
        fetch_raw,
        load_panel_output,
        load_raw_outputs,
        save_panel_outputs,
        save_raw_outputs,
    )
    from quant_research.data.cleaning import filter_raw_to_calendar, get_trading_calendar
    from quant_research.features.daily_return import add_daily_return
    from quant_research.features.dividend_growth import add_dividend_growth
    from quant_research.features.equity_vol_correlation import add_equity_vol_correlation
    from quant_research.features.news_sentiment import add_news_sentiment
    from quant_research.features.rate_momentum import add_rate_momentum
    from quant_research.features.rolling_volatility import add_rolling_volatility
except ImportError:  # allow direct execution without module context
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from quant_research.data.sources.fred import DEFAULT_TREASURY_SERIES
    from quant_research.data.sources.market_panel import (
        PanelConfig,
        build_panel,
        fetch_raw,
        load_panel_output,
        load_raw_outputs,
        save_panel_outputs,
        save_raw_outputs,
    )
    from quant_research.data.cleaning import filter_raw_to_calendar, get_trading_calendar
    from quant_research.features.daily_return import add_daily_return
    from quant_research.features.dividend_growth import add_dividend_growth
    from quant_research.features.equity_vol_correlation import add_equity_vol_correlation
    from quant_research.features.news_sentiment import add_news_sentiment
    from quant_research.features.rate_momentum import add_rate_momentum
    from quant_research.features.rolling_volatility import add_rolling_volatility


def _load_dotenv(dotenv_path: Path) -> None:
    """Minimal .env loader (no external deps)."""
    if not dotenv_path.exists() or not dotenv_path.is_file():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Market panel pipeline: fetch an equity, vol index, and US Treasury yields"
    )
    p.add_argument(
        "--mode",
        choices=("all", "fetch", "panel", "features"),
        default="all",
        help="Run mode: all=fetch+panel+features; fetch=download raw only; panel=build panel only; features=compute features only",
    )
    p.add_argument(
        "--refresh-raw",
        action="store_true",
        help="Force re-download and overwrite cached raw data (default: reuse cache if present)",
    )
    p.add_argument("--start", default="2018-01-01", help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", default="2024-12-31", help="End date (YYYY-MM-DD)")
    p.add_argument("--equity", default="JPM", help="Equity ticker (default: JPM)")
    p.add_argument("--vol", default="^VIX", help="Vol index ticker for Yahoo (default: ^VIX)")
    p.add_argument(
        "--vol-source",
        choices=("fred", "yahoo"),
        default="fred",
        help="Vol source (default: fred via VIXCLS)",
    )
    p.add_argument(
        "--vol-fred-series-id",
        default="VIXCLS",
        help="FRED series id for vol when --vol-source=fred (default: VIXCLS)",
    )
    p.add_argument(
        "--treasury-series",
        default=",".join(DEFAULT_TREASURY_SERIES),
        help="Comma-separated FRED series ids (default: common DGS* set)",
    )
    p.add_argument(
        "--fred-api-key",
        default=os.getenv("FRED_API_KEY", ""),
        help="FRED API key (recommended; or set env FRED_API_KEY)",
    )
    p.add_argument(
        "--alpha-vantage-api-key",
        default=os.getenv("ALPHAVANTAGE_API_KEY", ""),
        help="Alpha Vantage API key (or set env ALPHAVANTAGE_API_KEY)",
    )
    p.add_argument(
        "--prefer-alpha-vantage-for-equity",
        action="store_true",
        help="Use Alpha Vantage for equity daily (otherwise use Yahoo/yfinance)",
    )
    p.add_argument(
        "--include-options",
        action="store_true",
        help="Also fetch US equity options history via Alpha Vantage and write aggregated fields into panel",
    )
    p.add_argument(
        "--options-symbol",
        default="",
        help="Options underlying symbol for Alpha Vantage HISTORICAL_OPTIONS (default: same as --equity)",
    )
    p.add_argument(
        "--options-min-interval-secs",
        type=float,
        default=0.85,
        help="Minimum interval between Alpha Vantage options API calls (default: 0.85)",
    )
    p.add_argument(
        "--options-timeout-secs",
        type=int,
        default=60,
        help="Per-request timeout for options API (default: 60)",
    )
    p.add_argument(
        "--options-max-retries",
        type=int,
        default=6,
        help="Max retries per day for options API (default: 6)",
    )
    p.add_argument(
        "--options-progress-every",
        type=int,
        default=50,
        help="Print options progress every N requested days (default: 50)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output directory (default: <project_root>/output/market_panel)",
    )
    return p.parse_args()


def main() -> int:
    project_root = Path(__file__).resolve().parents[4]
    _load_dotenv(project_root / ".env")

    args = _parse_args()
    out_dir = (
        Path(args.out).expanduser().resolve()
        if args.out
        else (project_root / "output" / "market_panel")
    )

    treasury_series = tuple(s.strip() for s in str(args.treasury_series).split(",") if s.strip())
    cfg = PanelConfig(
        start=args.start,
        end=args.end,
        equity_ticker=args.equity,
        vol_ticker=args.vol,
        vol_source=str(args.vol_source),
        vol_fred_series_id=str(args.vol_fred_series_id),
        treasury_series=treasury_series or DEFAULT_TREASURY_SERIES,
        fred_api_key=args.fred_api_key or None,
        alpha_vantage_api_key=args.alpha_vantage_api_key or None,
        prefer_alpha_vantage_for_equity=bool(args.prefer_alpha_vantage_for_equity),
        include_options=bool(args.include_options),
        options_symbol=(str(args.options_symbol).strip() or None),
        options_min_interval_secs=float(args.options_min_interval_secs),
        options_timeout_secs=int(args.options_timeout_secs),
        options_max_retries=int(args.options_max_retries),
        options_progress_every=int(args.options_progress_every),
    )

    raw = None
    panel = None

    raw_dir = out_dir / "raw"
    has_raw_cache = (
        (raw_dir / "equity_daily.csv").exists()
        or (raw_dir / "vol_daily.csv").exists()
        or (raw_dir / "treasury_yields.csv").exists()
    )

    if args.mode in ("all", "fetch"):
        if has_raw_cache and not args.refresh_raw:
            print(f"[INFO] Using cached raw data under: {raw_dir} (pass --refresh-raw to refetch)")
            raw = load_raw_outputs(out_dir)
        else:
            raw = fetch_raw(cfg)
            save_raw_outputs(out_dir, raw, overwrite=bool(args.refresh_raw))

    if args.mode in ("all", "panel"):
        if raw is None:
            raw = load_raw_outputs(out_dir)
        trading_calendar = get_trading_calendar(raw, calendar_key="equity")
        raw = filter_raw_to_calendar(raw, trading_calendar)
        panel = build_panel(raw).ffill()
        save_panel_outputs(out_dir, panel)

    if args.mode in ("all", "features"):
        if panel is None:
            panel = load_panel_output(out_dir)
        if panel is None or panel.empty:
            print("Panel is empty or missing. Run with --mode panel (or --mode all) first.")
            return 1
        if raw is None:
            raw = load_raw_outputs(out_dir)

        panel = add_daily_return(panel)
        panel = add_rolling_volatility(panel, windows=(21, 63, 252))
        panel = add_dividend_growth(panel, (raw or {}).get("equity"), fill_method="ffill")
        panel = add_equity_vol_correlation(panel, windows=(21, 63, 252))
        panel = add_rate_momentum(panel, rate_col_candidates=("DGS10",), horizons=(21, 63), zscore_window=252)
        sentiment_path = out_dir / "news" / "sentiment_daily.csv"
        if sentiment_path.exists():
            panel = add_news_sentiment(panel, sentiment_csv_path=sentiment_path, fill_method="ffill")
        else:
            print(f"[INFO] News sentiment file not found; skip: {sentiment_path}")
        save_panel_outputs(out_dir, panel)

    print(f"Wrote: {out_dir}")
    if raw is None:
        raw = load_raw_outputs(out_dir)
    for name in ("equity", "vol", "treasury", "options"):
        df = (raw or {}).get(name)
        shape = getattr(df, "shape", None)
        print(f"{name}: {shape}")
    if panel is None:
        panel = load_panel_output(out_dir)
    if panel is not None and not panel.empty:
        print(f"Panel shape: {panel.shape}")
        print(f"Columns: {list(panel.columns)[:20]}{' ...' if len(panel.columns) > 20 else ''}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
