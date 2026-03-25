#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare Week4 baseline performance under different risk-free rate sources, "
            "with fixed sigma construction and liquidity filters."
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
        help="Panel CSV (default: output/market_panel/processed/panel.csv)",
    )
    p.add_argument(
        "--equity-csv",
        default="output/market_panel/raw/equity_daily.csv",
        help="Equity daily CSV with Dividend column (default: output/market_panel/raw/equity_daily.csv)",
    )
    p.add_argument(
        "--out-csv",
        default="output/week4_baseline/r_source_comparison.csv",
        help="Output CSV (default: output/week4_baseline/r_source_comparison.csv)",
    )
    p.add_argument("--symbol", default="JPM", help="Underlying symbol filter (default: JPM)")
    p.add_argument(
        "--sigma-method",
        choices=("blend_iv_rv252_75_25", "iv", "rv252"),
        default="blend_iv_rv252_75_25",
        help="Sigma method (default: blend_iv_rv252_75_25)",
    )
    p.add_argument(
        "--q-method",
        choices=("zero", "trailing_div_12m"),
        default="trailing_div_12m",
        help="Dividend yield method (default: trailing_div_12m)",
    )
    p.add_argument("--min-volume", type=float, default=10.0, help="Volume filter (default: 10)")
    p.add_argument("--min-open-interest", type=float, default=0.0, help="Open interest filter (default: 0)")
    p.add_argument(
        "--r-sources",
        default="DGS3MO,DGS6MO,DGS1,DGS2,DGS5,DGS7,DGS10",
        help="Comma-separated panel rate columns to test (default: DGS3MO,DGS6MO,DGS1,DGS2,DGS5,DGS7,DGS10)",
    )
    return p.parse_args()


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_price(*, is_call: bool, s: float, k: float, t: float, r: float, q: float, sigma: float) -> float:
    if t <= 0:
        return max(0.0, (s - k) if is_call else (k - s))
    if sigma <= 0:
        fwd = s * math.exp((r - q) * t)
        disc = math.exp(-r * t)
        intrinsic = max(0.0, (fwd - k) if is_call else (k - fwd))
        return disc * intrinsic
    sqrt_t = math.sqrt(t)
    d1 = (math.log(s / k) + (r - q + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    df_r = math.exp(-r * t)
    df_q = math.exp(-q * t)
    if is_call:
        return s * df_q * _norm_cdf(d1) - k * df_r * _norm_cdf(d2)
    return k * df_r * _norm_cdf(-d2) - s * df_q * _norm_cdf(-d1)


def _to_decimal_rate(s: Any) -> float:
    try:
        v = float(s)
    except Exception:
        return 0.0
    if not math.isfinite(v):
        return 0.0
    return float(v / 100.0) if abs(v) > 3.0 else float(v)


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
    for p in (options_csv, panel_csv):
        if not p.exists():
            print(f"fatal: missing file: {p}")
            return 2

    opt = pd.read_csv(options_csv)
    panel = pd.read_csv(panel_csv)
    if opt.empty or panel.empty:
        print("fatal: options/panel is empty")
        return 2

    # Options normalize
    if "date" in opt.columns:
        opt["date"] = pd.to_datetime(opt["date"], errors="coerce").dt.normalize()
    else:
        first = str(opt.columns[0])
        opt["date"] = pd.to_datetime(opt[first], errors="coerce").dt.normalize()
    opt = opt.dropna(subset=["date"]).copy()

    if "symbol" in opt.columns:
        sym = str(args.symbol).strip().upper()
        opt = opt[opt["symbol"].astype(str).str.upper() == sym]
    if opt.empty:
        print("fatal: no options rows after symbol filter")
        return 2

    for c in ("bid", "ask", "last", "mark", "strike", "implied_volatility", "volume", "open_interest"):
        if c in opt.columns:
            opt[c] = pd.to_numeric(opt[c], errors="coerce")
    opt["expiration"] = pd.to_datetime(opt.get("expiration"), errors="coerce").dt.normalize()
    opt["option_type"] = opt.get("type", "call").astype(str).str.lower()

    # market_price = MID ONLY
    bid = pd.to_numeric(opt.get("bid"), errors="coerce")
    ask = pd.to_numeric(opt.get("ask"), errors="coerce")
    mid = (bid + ask) / 2.0
    valid_mid = bid.notna() & ask.notna() & (bid >= 0) & (ask >= bid)
    opt["market_price"] = mid.where(valid_mid, np.nan)

    # Panel normalize
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    need = ["date", "equity_close", "equity_rvol_252"]
    r_sources = [s.strip() for s in str(args.r_sources).split(",") if s.strip()]
    for r in r_sources:
        if r in panel.columns:
            need.append(r)
    panel = panel[[c for c in need if c in panel.columns]].drop_duplicates(subset=["date"], keep="last")
    for c in ("equity_close", "equity_rvol_252", *r_sources):
        if c in panel.columns:
            panel[c] = pd.to_numeric(panel[c], errors="coerce")

    df = opt.merge(panel, on="date", how="left")
    df["S0"] = pd.to_numeric(df["equity_close"], errors="coerce")
    df["strike_price"] = pd.to_numeric(df["strike"], errors="coerce")
    df["iv"] = pd.to_numeric(df["implied_volatility"], errors="coerce")
    # IV is already decimal in AV options payload. Guard for accidental percent.
    df["iv"] = np.where(df["iv"] > 3.0, df["iv"] / 100.0, df["iv"])
    df["rv252"] = pd.to_numeric(df.get("equity_rvol_252"), errors="coerce")

    if str(args.sigma_method) == "iv":
        df["sigma"] = df["iv"]
    elif str(args.sigma_method) == "rv252":
        df["sigma"] = df["rv252"]
    else:
        # blend_iv_rv252_75_25
        df["sigma"] = 0.75 * df["iv"] + 0.25 * df["rv252"]

    # q construction
    df["q"] = 0.0
    if str(args.q_method) == "trailing_div_12m" and equity_csv.exists():
        eq = pd.read_csv(equity_csv)
        if "date" in eq.columns:
            eq["date"] = pd.to_datetime(eq["date"], errors="coerce").dt.normalize()
        else:
            first = str(eq.columns[0])
            eq["date"] = pd.to_datetime(eq[first], errors="coerce").dt.normalize()
        if "Dividend" in eq.columns:
            eq = eq.dropna(subset=["date"]).copy()
            eq["Dividend"] = pd.to_numeric(eq["Dividend"], errors="coerce").fillna(0.0)
            eq = eq.sort_values("date").set_index("date")
            div_12m = eq["Dividend"].rolling("365D", min_periods=1).sum()
            q_df = div_12m.reset_index().rename(columns={"Dividend": "div_12m"})
            df = df.merge(q_df, on="date", how="left")
            df["q"] = (pd.to_numeric(df["div_12m"], errors="coerce") / df["S0"]).replace([np.inf, -np.inf], np.nan)
            df["q"] = df["q"].fillna(0.0).clip(lower=0.0, upper=0.25)

    dte = (df["expiration"] - df["date"]).dt.days
    df["T_years"] = (dte.clip(lower=0) / 365.0).astype(float)

    # Liquidity filters
    if "volume" in df.columns and float(args.min_volume) > 0:
        df = df[df["volume"] >= float(args.min_volume)]
    if "open_interest" in df.columns and float(args.min_open_interest) > 0:
        df = df[df["open_interest"] >= float(args.min_open_interest)]

    base = df.dropna(subset=["market_price", "S0", "strike_price", "T_years", "sigma"]).copy()
    base = base[(base["market_price"] > 0) & (base["S0"] > 0) & (base["strike_price"] > 0) & (base["sigma"] > 0)]
    if base.empty:
        print("fatal: no rows after filters")
        return 2

    results: list[dict[str, Any]] = []
    for r_col in r_sources:
        if r_col not in base.columns:
            continue
        tmp = base.dropna(subset=[r_col]).copy()
        if tmp.empty:
            continue
        tmp["r"] = tmp[r_col].map(_to_decimal_rate)
        pred = []
        for _, row in tmp.iterrows():
            pred.append(
                _bs_price(
                    is_call=("c" in str(row["option_type"]).lower()),
                    s=float(row["S0"]),
                    k=float(row["strike_price"]),
                    t=max(0.0, float(row["T_years"])),
                    r=float(row["r"]),
                    q=float(row["q"]),
                    sigma=float(row["sigma"]),
                )
            )
        import numpy as np

        yhat = np.array(pred, dtype=float)
        y = tmp["market_price"].to_numpy(dtype=float)
        e = yhat - y
        mae = float(np.mean(np.abs(e)))
        rmse = float(np.sqrt(np.mean(e * e)))
        results.append(
            {
                "r_source": r_col,
                "sigma_method": str(args.sigma_method),
                "market_price_method": "mid_only",
                "min_volume": float(args.min_volume),
                "min_open_interest": float(args.min_open_interest),
                "q_method": str(args.q_method),
                "n": int(len(tmp)),
                "mae": mae,
                "rmse": rmse,
            }
        )

    if not results:
        print("fatal: no valid r_source results")
        return 2

    out = pd.DataFrame(results).sort_values(["mae", "rmse"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"saved: {out_csv}")
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
