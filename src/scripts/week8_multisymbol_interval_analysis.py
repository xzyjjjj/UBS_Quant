#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from quant_research.data.sources.alpha_vantage import (
        fetch_alpha_vantage_daily_adjusted,
        fetch_alpha_vantage_historical_options,
    )
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from quant_research.data.sources.alpha_vantage import (
        fetch_alpha_vantage_daily_adjusted,
        fetch_alpha_vantage_historical_options,
    )


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(
        description="Week8 multi-symbol interval study: fetch AV options, run backtest, analyze weekday/month effects."
    )
    p.add_argument("--symbols", default="JPM,AAPL,MSFT,NVDA,TSLA")
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--lookback-days", type=int, default=252)
    p.add_argument("--target-dte", type=int, default=14)
    p.add_argument("--dte-tol", type=int, default=10)
    p.add_argument("--max-strike-gap", type=float, default=15.0)
    p.add_argument("--max-moneyness-dev", type=float, default=0.30)
    p.add_argument("--min-volume", type=float, default=1.0)
    p.add_argument("--min-open-interest", type=float, default=1.0)
    p.add_argument("--min-interval-secs", type=float, default=0.2)
    p.add_argument("--options-timeout-secs", type=int, default=60)
    p.add_argument("--options-max-retries", type=int, default=6)
    p.add_argument("--options-progress-every", type=int, default=50)
    p.add_argument("--refresh-fetch", action="store_true")
    p.add_argument("--refresh-backtest", action="store_true")
    p.add_argument(
        "--out-dir",
        default=str(root / "output" / "week8_multisymbol"),
    )
    return p.parse_args()


def load_dotenv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k:
            out[k] = v
            os.environ.setdefault(k, v)
    return out


def read_rate_series(project_root: Path) -> pd.Series:
    candidates = [
        project_root / "output" / "market_panel" / "raw" / "treasury_yields.csv",
        project_root / "output" / "server_bundle_week5" / "data" / "raw" / "treasury_yields.csv",
    ]
    for p in candidates:
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if "date" not in df.columns:
            continue
        idx = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        if "DGS10" not in df.columns:
            continue
        r = pd.to_numeric(df["DGS10"], errors="coerce")
        s = pd.Series(r.values, index=idx).sort_index()
        s = s[~s.index.isna()]
        s = s[~s.index.duplicated(keep="last")]
        return s.ffill()
    raise FileNotFoundError("Cannot find treasury_yields.csv with DGS10.")


def choose_market_price(df: pd.DataFrame) -> pd.Series:
    mark = pd.to_numeric(df.get("mark"), errors="coerce")
    last = pd.to_numeric(df.get("last"), errors="coerce")
    bid = pd.to_numeric(df.get("bid"), errors="coerce")
    ask = pd.to_numeric(df.get("ask"), errors="coerce")
    mid = (bid + ask) / 2.0
    px = mark.where(mark > 0, np.nan)
    px = px.fillna(last.where(last > 0, np.nan))
    px = px.fillna(mid.where(mid > 0, np.nan))
    return px


def normalize_sigma(sigma: pd.Series) -> pd.Series:
    x = pd.to_numeric(sigma, errors="coerce")
    x = x.where(x >= 0, np.nan)
    x = x.where(x <= 3.0, x / 100.0)
    return x.fillna(0.0)


@dataclass
class SymbolArtifacts:
    symbol: str
    dataset_csv: Path
    equity_csv: Path
    backtest_csv: Path


def build_symbol_dataset(
    *,
    symbol: str,
    start: str,
    end: str,
    api_key: str,
    rate_series: pd.Series,
    out_data_dir: Path,
    min_interval_secs: float,
    timeout_secs: int,
    max_retries: int,
    progress_every: int,
    refresh_fetch: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    symbol = symbol.upper()
    out_data_dir.mkdir(parents=True, exist_ok=True)
    dataset_csv = out_data_dir / f"{symbol}_options_final_av_{start}_{end}.csv"
    equity_csv = out_data_dir / f"{symbol}_equity_daily_{start}_{end}.csv"

    if dataset_csv.exists() and equity_csv.exists() and not refresh_fetch:
        ds = pd.read_csv(dataset_csv, parse_dates=["date", "expiration"])
        eq = pd.read_csv(equity_csv, parse_dates=["date"])
        return ds, eq, dataset_csv, equity_csv

    eq = fetch_alpha_vantage_daily_adjusted(symbol, start, end, api_key=api_key).reset_index()
    if eq.empty:
        raise RuntimeError(f"equity fetch returned empty for {symbol}")
    eq["date"] = pd.to_datetime(eq["date"], errors="coerce").dt.normalize()
    eq["S0"] = pd.to_numeric(eq.get("Close"), errors="coerce")
    eq = eq[["date", "S0"]].dropna().sort_values("date").drop_duplicates("date", keep="last")
    eq.to_csv(equity_csv, index=False)

    opt = fetch_alpha_vantage_historical_options(
        symbol,
        start,
        end,
        api_key=api_key,
        min_interval_secs=min_interval_secs,
        timeout=timeout_secs,
        max_retries=max_retries,
        progress_every=progress_every,
    )
    if opt.empty:
        raise RuntimeError(f"options fetch returned empty for {symbol}")
    opt = opt.reset_index()
    opt["date"] = pd.to_datetime(opt["date"], errors="coerce").dt.normalize()
    opt["expiration"] = pd.to_datetime(opt.get("expiration"), errors="coerce").dt.normalize()
    opt["option_type"] = opt.get("type", "").astype(str).str.lower()
    opt["strike_price"] = pd.to_numeric(opt.get("strike"), errors="coerce")
    opt["market_price"] = choose_market_price(opt)
    opt["volume"] = pd.to_numeric(opt.get("volume"), errors="coerce").fillna(0.0)
    opt["open_interest"] = pd.to_numeric(opt.get("open_interest"), errors="coerce").fillna(0.0)
    opt["sigma"] = normalize_sigma(opt.get("implied_volatility"))
    opt["contractID"] = opt.get("contractID", "").astype(str)
    missing_id = opt["contractID"].str.len() == 0
    if missing_id.any():
        sid = (
            symbol
            + "_"
            + opt.loc[missing_id, "date"].dt.strftime("%Y%m%d").fillna("00000000")
            + "_"
            + opt.loc[missing_id, "option_type"].fillna("na")
            + "_"
            + opt.loc[missing_id, "strike_price"].round(3).astype(str)
        )
        opt.loc[missing_id, "contractID"] = sid

    df = opt.merge(eq, on="date", how="left")
    r_map = rate_series.reindex(pd.to_datetime(df["date"], errors="coerce")).ffill().bfill()
    df["r"] = pd.Series(pd.to_numeric(r_map.values, errors="coerce"), index=df.index).fillna(0.0)
    df["q"] = 0.0
    t_days = (df["expiration"] - df["date"]).dt.days
    df["T_years"] = t_days.clip(lower=0).astype(float) / 365.0
    df["symbol"] = symbol

    keep = [
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
    df = df[keep].copy()
    df = df[df["option_type"].isin(["call", "put"])]
    df = df.dropna(subset=["date", "expiration", "strike_price", "market_price", "S0"])
    df = df[df["expiration"] > df["date"]]
    df = df[df["strike_price"] > 0]
    df = df[df["market_price"] > 0]
    df = df[df["S0"] > 0]
    df = df.sort_values(["date", "expiration", "contractID"]).reset_index(drop=True)
    df.to_csv(dataset_csv, index=False)
    return df, eq, dataset_csv, equity_csv


def run_backtest(
    *,
    project_root: Path,
    data_csv: Path,
    out_csv: Path,
    symbol: str,
    args: argparse.Namespace,
    refresh_backtest: bool,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists() and not refresh_backtest:
        return
    cmd = [
        sys.executable,
        str(project_root / "src" / "scripts" / "backtest_10d_7d_near_spot.py"),
        "--data-csv",
        str(data_csv),
        "--symbol",
        symbol.upper(),
        "--option-type",
        "call",
        "--lookback-days",
        str(int(args.lookback_days)),
        "--target-dte",
        str(int(args.target_dte)),
        "--dte-tol",
        str(int(args.dte_tol)),
        "--max-strike-gap",
        str(float(args.max_strike_gap)),
        "--max-moneyness-dev",
        str(float(args.max_moneyness_dev)),
        "--min-volume",
        str(float(args.min_volume)),
        "--min-open-interest",
        str(float(args.min_open_interest)),
        "--out-csv",
        str(out_csv),
    ]
    subprocess.run(cmd, check=True)


def month_return_map(equity_df: pd.DataFrame) -> pd.DataFrame:
    d = equity_df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date", "S0"]).sort_values("date")
    d["month"] = d["date"].dt.to_period("M").astype(str)
    g = d.groupby("month")["S0"]
    out = pd.DataFrame(
        {
            "month": sorted(g.groups.keys()),
            "close_first": g.first().values,
            "close_last": g.last().values,
        }
    )
    out["equity_ret"] = out["close_last"] / out["close_first"] - 1.0
    return out


def to_ordered_weekday(series: pd.Series) -> pd.Categorical:
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    return pd.Categorical(series, categories=order, ordered=True)


def summarize_symbol(
    symbol: str,
    backtest_csv: Path,
    equity_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    raw = pd.read_csv(backtest_csv)
    raw["trade_date"] = pd.to_datetime(raw["trade_date"], errors="coerce")
    raw["expiration"] = pd.to_datetime(raw.get("expiration"), errors="coerce")
    raw["spot_trade"] = pd.to_numeric(raw.get("spot_trade"), errors="coerce")
    raw["strike"] = pd.to_numeric(raw.get("strike"), errors="coerce")
    raw["premium"] = pd.to_numeric(raw.get("premium"), errors="coerce")
    raw["intrinsic_expiry"] = pd.to_numeric(raw.get("intrinsic_expiry"), errors="coerce")
    raw["pnl_1lot"] = pd.to_numeric(raw.get("pnl_1lot"), errors="coerce")
    raw["weekday"] = raw["trade_date"].dt.day_name()
    raw["month"] = raw["trade_date"].dt.to_period("M").astype(str)
    raw["quarter"] = raw["trade_date"].dt.to_period("Q").astype(str)
    raw["dte"] = (raw["expiration"] - raw["trade_date"]).dt.days
    raw["moneyness"] = raw["spot_trade"] / raw["strike"]

    ex = raw[raw["status"] == "EXECUTED"].copy()
    ex["symbol"] = symbol
    ex["win"] = (ex["pnl_1lot"] > 0).astype(float)
    ex["positive_intrinsic"] = (ex["intrinsic_expiry"] > 0).astype(float)

    overall = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "days_total": int(len(raw)),
                "days_executed": int(len(ex)),
                "exec_rate": float(len(ex) / len(raw)) if len(raw) > 0 else 0.0,
                "win_rate": float(ex["win"].mean()) if len(ex) > 0 else np.nan,
                "pnl_sum": float(ex["pnl_1lot"].sum()) if len(ex) > 0 else 0.0,
                "avg_pnl": float(ex["pnl_1lot"].mean()) if len(ex) > 0 else np.nan,
            }
        ]
    )

    weekday = (
        ex.groupby("weekday", dropna=False)
        .agg(
            trades=("win", "size"),
            win_rate=("win", "mean"),
            pnl_sum=("pnl_1lot", "sum"),
            avg_pnl=("pnl_1lot", "mean"),
            median_pnl=("pnl_1lot", "median"),
            avg_moneyness=("moneyness", "mean"),
            avg_dte=("dte", "mean"),
            positive_intrinsic_rate=("positive_intrinsic", "mean"),
            avg_premium=("premium", "mean"),
        )
        .reset_index()
    )
    weekday["weekday"] = to_ordered_weekday(weekday["weekday"])
    weekday = weekday.sort_values("weekday")
    weekday["symbol"] = symbol

    monthly = (
        ex.groupby("month", dropna=False)
        .agg(
            trades=("win", "size"),
            win_rate=("win", "mean"),
            pnl_sum=("pnl_1lot", "sum"),
            avg_pnl=("pnl_1lot", "mean"),
            avg_moneyness=("moneyness", "mean"),
            avg_dte=("dte", "mean"),
            positive_intrinsic_rate=("positive_intrinsic", "mean"),
            avg_premium=("premium", "mean"),
        )
        .reset_index()
        .sort_values("month")
    )
    monthly["symbol"] = symbol
    eq_month = month_return_map(equity_df)
    monthly = monthly.merge(eq_month[["month", "equity_ret"]], on="month", how="left")

    interval_rows: list[dict[str, object]] = []
    interval_defs = [
        ("full_sample", "2024-01-01", "2024-12-31"),
        ("q1_2024", "2024-01-01", "2024-03-31"),
        ("q2_2024", "2024-04-01", "2024-06-30"),
        ("q3_2024", "2024-07-01", "2024-09-30"),
        ("q4_2024", "2024-10-01", "2024-12-31"),
    ]
    for name, s, e in interval_defs:
        m = ex[(ex["trade_date"] >= pd.Timestamp(s)) & (ex["trade_date"] <= pd.Timestamp(e))]
        fri = m[m["weekday"] == "Friday"]
        tue = m[m["weekday"] == "Tuesday"]
        jan = m[m["trade_date"].dt.month == 1]
        mar = m[m["trade_date"].dt.month == 3]
        interval_rows.append(
            {
                "symbol": symbol,
                "interval": name,
                "trades_total": int(len(m)),
                "fri_trades": int(len(fri)),
                "fri_win_rate": float((fri["pnl_1lot"] > 0).mean()) if len(fri) > 0 else np.nan,
                "fri_pnl_sum": float(fri["pnl_1lot"].sum()) if len(fri) > 0 else 0.0,
                "tue_trades": int(len(tue)),
                "tue_win_rate": float((tue["pnl_1lot"] > 0).mean()) if len(tue) > 0 else np.nan,
                "tue_pnl_sum": float(tue["pnl_1lot"].sum()) if len(tue) > 0 else 0.0,
                "fri_minus_tue_pnl": float(fri["pnl_1lot"].sum() - tue["pnl_1lot"].sum()),
                "fri_minus_tue_winrate": (
                    float((fri["pnl_1lot"] > 0).mean() - (tue["pnl_1lot"] > 0).mean())
                    if (len(fri) > 0 and len(tue) > 0)
                    else np.nan
                ),
                "jan_trades": int(len(jan)),
                "jan_win_rate": float((jan["pnl_1lot"] > 0).mean()) if len(jan) > 0 else np.nan,
                "jan_pnl_sum": float(jan["pnl_1lot"].sum()) if len(jan) > 0 else 0.0,
                "mar_trades": int(len(mar)),
                "mar_win_rate": float((mar["pnl_1lot"] > 0).mean()) if len(mar) > 0 else np.nan,
                "mar_pnl_sum": float(mar["pnl_1lot"].sum()) if len(mar) > 0 else 0.0,
                "mar_minus_jan_winrate": (
                    float((mar["pnl_1lot"] > 0).mean() - (jan["pnl_1lot"] > 0).mean())
                    if (len(mar) > 0 and len(jan) > 0)
                    else np.nan
                ),
                "mar_minus_jan_pnl": float(mar["pnl_1lot"].sum() - jan["pnl_1lot"].sum()),
            }
        )
    interval = pd.DataFrame(interval_rows)

    jan_mar = ex[ex["trade_date"].dt.month.isin([1, 3])].copy()
    jan_mar["month_name"] = jan_mar["trade_date"].dt.month.map({1: "January", 3: "March"})
    jan_mar_detail = (
        jan_mar.groupby("month_name")
        .agg(
            trades=("win", "size"),
            win_rate=("win", "mean"),
            pnl_sum=("pnl_1lot", "sum"),
            avg_pnl=("pnl_1lot", "mean"),
            avg_moneyness=("moneyness", "mean"),
            avg_dte=("dte", "mean"),
            positive_intrinsic_rate=("positive_intrinsic", "mean"),
            avg_premium=("premium", "mean"),
        )
        .reset_index()
    )
    jan_mar_detail["symbol"] = symbol
    jan_mar_detail = jan_mar_detail.merge(
        eq_month.rename(columns={"month": "month_key"}),
        left_on=jan_mar_detail["month_name"].map({"January": "2024-01", "March": "2024-03"}),
        right_on="month_key",
        how="left",
    ).drop(columns=["key_0", "month_key"], errors="ignore")

    return {
        "overall": overall,
        "weekday": weekday,
        "monthly": monthly,
        "interval": interval,
        "jan_mar_detail": jan_mar_detail,
    }


def concat_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    valid = [f for f in frames if f is not None and not f.empty]
    if not valid:
        return pd.DataFrame()
    return pd.concat(valid, ignore_index=True)


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    out_dir = Path(args.out_dir).expanduser().resolve()
    data_dir = out_dir / "data"
    bt_dir = out_dir / "backtest"
    analysis_dir = out_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    env = load_dotenv(project_root / ".env")
    api_key = (os.getenv("ALPHAVANTAGE_API_KEY") or env.get("ALPHAVANTAGE_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("Missing ALPHAVANTAGE_API_KEY.")

    rate_series = read_rate_series(project_root)
    symbols = [s.strip().upper() for s in str(args.symbols).split(",") if s.strip()]
    if len(symbols) < 1:
        raise RuntimeError("No symbols provided.")

    artifacts: list[SymbolArtifacts] = []
    summaries: dict[str, list[pd.DataFrame]] = {
        "overall": [],
        "weekday": [],
        "monthly": [],
        "interval": [],
        "jan_mar_detail": [],
    }

    for i, symbol in enumerate(symbols, start=1):
        print(f"[INFO] ({i}/{len(symbols)}) prepare symbol={symbol}")
        ds, eq, ds_csv, eq_csv = build_symbol_dataset(
            symbol=symbol,
            start=str(args.start),
            end=str(args.end),
            api_key=api_key,
            rate_series=rate_series,
            out_data_dir=data_dir,
            min_interval_secs=float(args.min_interval_secs),
            timeout_secs=int(args.options_timeout_secs),
            max_retries=int(args.options_max_retries),
            progress_every=int(args.options_progress_every),
            refresh_fetch=bool(args.refresh_fetch),
        )
        bt_csv = bt_dir / f"{symbol}_backtest_{args.start}_{args.end}.csv"
        print(f"[INFO] ({i}/{len(symbols)}) run backtest symbol={symbol}")
        run_backtest(
            project_root=project_root,
            data_csv=ds_csv,
            out_csv=bt_csv,
            symbol=symbol,
            args=args,
            refresh_backtest=bool(args.refresh_backtest),
        )
        artifacts.append(SymbolArtifacts(symbol=symbol, dataset_csv=ds_csv, equity_csv=eq_csv, backtest_csv=bt_csv))
        s = summarize_symbol(symbol, bt_csv, eq)
        for k in summaries:
            summaries[k].append(s[k])

    overall = concat_frames(summaries["overall"])
    weekday = concat_frames(summaries["weekday"])
    monthly = concat_frames(summaries["monthly"])
    interval = concat_frames(summaries["interval"])
    jan_mar_detail = concat_frames(summaries["jan_mar_detail"])

    overall.to_csv(analysis_dir / "overall_summary.csv", index=False)
    weekday.to_csv(analysis_dir / "weekday_summary_by_symbol.csv", index=False)
    monthly.to_csv(analysis_dir / "monthly_summary_by_symbol.csv", index=False)
    interval.to_csv(analysis_dir / "interval_friday_tuesday_jan_mar_summary.csv", index=False)
    jan_mar_detail.to_csv(analysis_dir / "jan_mar_driver_metrics_by_symbol.csv", index=False)

    overall_notes = []
    for a in artifacts:
        overall_notes.append(
            {
                "symbol": a.symbol,
                "dataset_csv": str(a.dataset_csv),
                "equity_csv": str(a.equity_csv),
                "backtest_csv": str(a.backtest_csv),
            }
        )
    pd.DataFrame(overall_notes).to_csv(analysis_dir / "artifact_manifest.csv", index=False)

    print("[OK] finished")
    print(f"[OK] out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
