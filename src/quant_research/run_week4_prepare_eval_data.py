#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Prepare Week4 baseline eval input from options history + market panel. "
            "Output columns are aligned with run_week4_baseline.py."
        )
    )
    p.add_argument(
        "--options-csv",
        default="output/market_panel/raw/options_history.csv",
        help="Raw options history CSV (default: output/market_panel/raw/options_history.csv)",
    )
    p.add_argument(
        "--panel-csv",
        default="output/market_panel/processed/panel.csv",
        help="Market panel CSV (default: output/market_panel/processed/panel.csv)",
    )
    p.add_argument(
        "--equity-csv",
        default="output/market_panel/raw/equity_daily.csv",
        help="Raw equity daily CSV with Dividend column (default: output/market_panel/raw/equity_daily.csv)",
    )
    p.add_argument(
        "--out-csv",
        default="output/cme/jpm_options.csv",
        help="Output eval CSV (default: output/cme/jpm_options.csv)",
    )
    p.add_argument(
        "--symbol",
        default="JPM",
        help="Filter options by symbol (default: JPM)",
    )
    p.add_argument(
        "--min-price",
        type=float,
        default=1e-6,
        help="Drop rows with market_price <= min-price (default: 1e-6)",
    )
    p.add_argument(
        "--q-method",
        choices=("zero", "trailing_div_12m"),
        default="trailing_div_12m",
        help="Dividend yield method for q (default: trailing_div_12m)",
    )
    p.add_argument(
        "--market-price-method",
        choices=("last_mid_mark", "mid_only"),
        default="mid_only",
        help="Market price construction (default: mid_only)",
    )
    p.add_argument(
        "--r-source",
        default="DGS3MO",
        help="Risk-free rate source column in panel (default: DGS3MO)",
    )
    p.add_argument(
        "--sigma-source",
        choices=("iv", "rv252", "blend_iv_rv252_75_25"),
        default="blend_iv_rv252_75_25",
        help="Sigma construction (default: blend_iv_rv252_75_25)",
    )
    return p.parse_args()


def _to_decimal_rate(x: object) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    if not math.isfinite(v):
        return 0.0
    return float(v / 100.0) if abs(v) > 3.0 else float(v)


def _to_decimal_sigma(x: object) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    if not math.isfinite(v):
        return float("nan")
    return float(v / 100.0) if v > 3.0 else float(v)


def main() -> int:
    args = _parse_args()
    try:
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore
    except Exception as e:
        print(f"fatal: missing dependency numpy/pandas: {e}")
        return 2

    options_csv = Path(args.options_csv)
    panel_csv = Path(args.panel_csv)
    equity_csv = Path(args.equity_csv)
    out_csv = Path(args.out_csv)
    if not options_csv.exists():
        print(f"fatal: missing options csv: {options_csv}")
        return 2
    if not panel_csv.exists():
        print(f"fatal: missing panel csv: {panel_csv}")
        return 2

    opt = pd.read_csv(options_csv)
    panel = pd.read_csv(panel_csv)
    if opt.empty:
        print("fatal: options csv is empty")
        return 2
    if panel.empty:
        print("fatal: panel csv is empty")
        return 2

    # Normalize options date.
    if "date" in opt.columns:
        opt["date"] = pd.to_datetime(opt["date"], errors="coerce").dt.normalize()
    else:
        # If index date was persisted as first unnamed column.
        first_col = str(opt.columns[0])
        if first_col.startswith("Unnamed"):
            opt["date"] = pd.to_datetime(opt[first_col], errors="coerce").dt.normalize()
        else:
            print("fatal: options csv missing date column")
            return 2
    opt = opt.dropna(subset=["date"]).copy()

    symbol = str(args.symbol).strip().upper()
    if "symbol" in opt.columns:
        opt = opt[opt["symbol"].astype(str).str.upper() == symbol].copy()
    if opt.empty:
        print(f"fatal: no options rows after symbol filter: {symbol}")
        return 2

    # Parse core options fields.
    for c in ("last", "mark", "bid", "ask", "strike", "implied_volatility", "volume", "open_interest"):
        if c in opt.columns:
            opt[c] = pd.to_numeric(opt[c], errors="coerce")
    if "expiration" not in opt.columns:
        print("fatal: options csv missing expiration column")
        return 2
    opt["expiration"] = pd.to_datetime(opt["expiration"], errors="coerce").dt.normalize()
    opt["option_type"] = opt.get("type", "call").astype(str).str.lower()

    # Market price proxy.
    last = pd.to_numeric(opt.get("last"), errors="coerce")
    bid = pd.to_numeric(opt.get("bid"), errors="coerce")
    ask = pd.to_numeric(opt.get("ask"), errors="coerce")
    mark = pd.to_numeric(opt.get("mark"), errors="coerce")
    mid = (bid + ask) / 2.0
    valid_mid = bid.notna() & ask.notna() & (bid >= 0) & (ask >= bid)
    if str(args.market_price_method) == "mid_only":
        market_price = np.where(valid_mid & (mid > 0), mid, np.nan)
    else:
        market_price = np.where(last > 0, last, np.where(valid_mid & (mid > 0), mid, mark))
    opt["market_price"] = pd.to_numeric(market_price, errors="coerce")

    # Sigma, strike.
    opt["sigma_iv"] = opt.get("implied_volatility").map(_to_decimal_sigma)
    opt["strike_price"] = pd.to_numeric(opt.get("strike"), errors="coerce")

    # Merge panel on date for spot and rates.
    if "date" not in panel.columns:
        first_col = str(panel.columns[0])
        if first_col.startswith("Unnamed"):
            panel["date"] = pd.to_datetime(panel[first_col], errors="coerce").dt.normalize()
        else:
            print("fatal: panel csv missing date column")
            return 2
    else:
        panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    panel = panel.dropna(subset=["date"]).copy()

    spot_col = "equity_close" if "equity_close" in panel.columns else ("equity_adj_close" if "equity_adj_close" in panel.columns else None)
    if spot_col is None:
        print("fatal: panel csv missing equity_close/equity_adj_close")
        return 2

    rate_col = str(args.r_source).strip()
    if rate_col not in panel.columns:
        # fallback if requested source missing
        rate_col = None
        for c in ("DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS5", "DGS7", "DGS10"):
            if c in panel.columns:
                rate_col = c
                break

    keep_panel_cols = ["date", spot_col]
    if rate_col:
        keep_panel_cols.append(rate_col)
    if "equity_rvol_252" in panel.columns:
        keep_panel_cols.append("equity_rvol_252")
    panel2 = panel[keep_panel_cols].drop_duplicates(subset=["date"], keep="last")

    df = opt.merge(panel2, on="date", how="left")
    df["S0"] = pd.to_numeric(df[spot_col], errors="coerce")
    df["r"] = df[rate_col].map(_to_decimal_rate) if rate_col else 0.0
    df["q"] = 0.0
    df["sigma_rv252"] = pd.to_numeric(df.get("equity_rvol_252"), errors="coerce")

    sigma_source = str(args.sigma_source)
    if sigma_source == "iv":
        df["sigma"] = df["sigma_iv"]
    elif sigma_source == "rv252":
        df["sigma"] = df["sigma_rv252"]
    else:
        # blend_iv_rv252_75_25
        df["sigma"] = 0.75 * df["sigma_iv"] + 0.25 * df["sigma_rv252"]

    if str(args.q_method) == "trailing_div_12m":
        if not equity_csv.exists():
            print(f"[WARN] q-method=trailing_div_12m but equity csv missing: {equity_csv}. Fallback q=0.")
        else:
            eq = pd.read_csv(equity_csv)
            # raw saved with index=True; typically first column is date.
            if "date" in eq.columns:
                eq["date"] = pd.to_datetime(eq["date"], errors="coerce").dt.normalize()
            else:
                first_col = str(eq.columns[0])
                eq["date"] = pd.to_datetime(eq[first_col], errors="coerce").dt.normalize()
            if "Dividend" not in eq.columns:
                print("[WARN] equity csv missing Dividend column. Fallback q=0.")
            else:
                eq = eq.dropna(subset=["date"]).copy()
                eq["Dividend"] = pd.to_numeric(eq["Dividend"], errors="coerce").fillna(0.0)
                # trailing 365D cash dividends / spot
                eq = eq.sort_values("date")
                eq = eq.set_index("date")
                div_12m = eq["Dividend"].rolling("365D", min_periods=1).sum()
                q_df = div_12m.reset_index().rename(columns={"Dividend": "div_12m"})
                q_df["date"] = pd.to_datetime(q_df["date"], errors="coerce").dt.normalize()
                q_df = q_df.drop_duplicates(subset=["date"], keep="last")
                df = df.merge(q_df, on="date", how="left")
                df["q"] = (pd.to_numeric(df["div_12m"], errors="coerce") / df["S0"]).replace([np.inf, -np.inf], np.nan)
                df["q"] = df["q"].fillna(0.0).clip(lower=0.0, upper=0.25)

    # Time-to-maturity in years.
    dte = (df["expiration"] - df["date"]).dt.days
    df["T_years"] = (dte.clip(lower=0) / 365.0).astype(float)

    out = df[
        [
            "date",
            "symbol",
            "option_type",
            "strike_price",
            "expiration",
            "market_price",
            "S0",
            "sigma",
            "r",
            "q",
            "T_years",
            "volume",
            "open_interest",
            "contractID",
        ]
    ].copy()
    out = out.dropna(subset=["market_price", "S0", "sigma", "strike_price", "T_years"])
    out = out[out["market_price"] > float(args.min_price)]
    out = out[out["S0"] > 0]
    out = out[out["sigma"] > 0]
    out = out[out["strike_price"] > 0]
    out = out.sort_values(["date", "expiration", "option_type", "strike_price"])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"saved: {out_csv}")
    print(
        f"rows={len(out)} date_range=({out['date'].min()}, {out['date'].max()}) symbol={symbol} "
        f"market_price_method={args.market_price_method} sigma_source={args.sigma_source} "
        f"r_source={(rate_col or 'fallback_none')} q_method={args.q_method}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
