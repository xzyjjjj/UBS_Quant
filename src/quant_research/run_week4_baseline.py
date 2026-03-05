#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Week4 baseline pipeline: evaluate BSM pricing with real traded option prices "
            "(MAE/RMSE), and locate failure regimes (high volatility / sentiment shift)."
        )
    )
    p.add_argument(
        "--cme-csv",
        default="output/cme/jpm_options.csv",
        help="CSV with real option traded prices (default: output/cme/jpm_options.csv)",
    )
    p.add_argument(
        "--panel-csv",
        default="output/market_panel/processed/panel.csv",
        help="Optional market panel CSV for volatility proxy (default: output/market_panel/processed/panel.csv)",
    )
    p.add_argument(
        "--sentiment-csv",
        default="output/news/sentiment_daily.csv",
        help="Optional sentiment CSV (default: output/news/sentiment_daily.csv)",
    )
    p.add_argument(
        "--out-dir",
        default="output/week4_baseline",
        help="Output directory (default: output/week4_baseline)",
    )
    p.add_argument(
        "--price-col",
        default="",
        help="Real market price column override in CME csv (default: auto detect)",
    )
    p.add_argument(
        "--bs-col",
        default="",
        help="BSM prediction column override in CME csv (default: auto detect or compute)",
    )
    p.add_argument(
        "--vol-quantile",
        type=float,
        default=0.8,
        help="Top quantile to define high-vol regime (default: 0.8)",
    )
    p.add_argument(
        "--sent-delta-quantile",
        type=float,
        default=0.8,
        help="Top quantile of abs sentiment change to define sentiment-shift regime (default: 0.8)",
    )
    p.add_argument(
        "--min-volume",
        type=float,
        default=0.0,
        help="Optional liquidity filter: keep rows with volume >= min-volume (default: 0, no extra filter)",
    )
    p.add_argument(
        "--min-open-interest",
        type=float,
        default=0.0,
        help="Optional liquidity filter: keep rows with open_interest >= min-open-interest (default: 0)",
    )
    p.add_argument(
        "--min-market-price",
        type=float,
        default=0.0,
        help="Optional filter: keep rows with market_price >= min-market-price (default: 0)",
    )
    return p.parse_args()


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        v = float(x)
        return v if math.isfinite(v) else None
    s = str(x).strip()
    if not s:
        return None
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except Exception:
        return None


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


def _pick_col(cols: list[str], candidates: tuple[str, ...]) -> str | None:
    lower_map = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def _build_date_col(df: Any) -> str:
    date_col = _pick_col(
        list(df.columns),
        (
            "date",
            "trade_date",
            "quote_date",
            "asof_date",
            "timestamp",
            "datetime",
            "update_time",
            "time_key",
        ),
    )
    if date_col is None:
        raise ValueError("No date-like column found in CME csv.")
    return date_col


def _normalize_rate(x: Any) -> float:
    v = _safe_float(x)
    if v is None:
        return 0.0
    if abs(v) > 3.0:
        return float(v / 100.0)
    return float(v)


def _normalize_sigma(x: Any) -> float:
    v = _safe_float(x)
    if v is None:
        return 0.0
    if v > 3.0:
        return float(v / 100.0)
    return float(v)


def _bool_is_call(x: Any) -> bool:
    s = str(x).strip().upper()
    return "C" in s or "CALL" in s


@dataclass(frozen=True)
class EvalResult:
    rows_input: int
    rows_eval: int
    market_price_col: str
    bs_price_col: str
    vol_proxy_col: str | None
    vol_threshold: float | None
    sent_delta_threshold: float | None


def _compute_metrics(df: Any, tag: str) -> dict[str, Any]:
    import numpy as np

    if df is None or len(df) == 0:
        return {"regime": tag, "n": 0, "mae": None, "rmse": None}
    err = df["abs_error"].to_numpy(dtype=float)
    sq = df["sq_error"].to_numpy(dtype=float)
    return {
        "regime": tag,
        "n": int(len(df)),
        "mae": float(np.nanmean(err)),
        "rmse": float(np.sqrt(np.nanmean(sq))),
    }


def _safe_qcut(s: Any, q: int, labels: list[str]) -> Any:
    import pandas as pd

    try:
        return pd.qcut(s, q=q, labels=labels, duplicates="drop")
    except Exception:
        return pd.Series([None] * len(s), index=s.index)


def _build_error_profile(eval_df: Any) -> Any:
    import numpy as np
    import pandas as pd

    df = eval_df.copy()
    # Moneyness: S/K
    if "S0" in df.columns and "strike_price" in df.columns:
        s = pd.to_numeric(df["S0"], errors="coerce")
        k = pd.to_numeric(df["strike_price"], errors="coerce")
        m = s / k
        df["moneyness"] = m
        df["moneyness_bucket"] = pd.cut(
            m,
            bins=[-np.inf, 0.9, 0.97, 1.03, 1.1, np.inf],
            labels=["deep_otm", "otm", "atm", "itm", "deep_itm"],
        )

    # T buckets (years)
    if "T_years" in df.columns:
        t = pd.to_numeric(df["T_years"], errors="coerce")
        df["ttm_bucket"] = pd.cut(
            t,
            bins=[-np.inf, 7 / 365.0, 30 / 365.0, 90 / 365.0, 180 / 365.0, np.inf],
            labels=["<=1w", "1w-1m", "1m-3m", "3m-6m", ">6m"],
        )

    # IV buckets (quantile)
    if "sigma" in df.columns:
        sig = pd.to_numeric(df["sigma"], errors="coerce")
        df["iv_bucket"] = _safe_qcut(sig, q=5, labels=["q1", "q2", "q3", "q4", "q5"])

    # Price buckets (quantile)
    px = pd.to_numeric(df["market_price"], errors="coerce")
    df["price_bucket"] = _safe_qcut(px, q=5, labels=["q1", "q2", "q3", "q4", "q5"])

    # Absolute error quantile flag (largest 10%)
    abs_err = pd.to_numeric(df["abs_error"], errors="coerce")
    if abs_err.dropna().size > 0:
        th = float(abs_err.quantile(0.9))
        df["is_large_error"] = abs_err >= th
    else:
        df["is_large_error"] = False

    group_cols: list[str] = []
    for c in (
        "option_type",
        "is_high_vol",
        "is_sent_shift",
        "moneyness_bucket",
        "ttm_bucket",
        "iv_bucket",
        "price_bucket",
    ):
        if c in df.columns:
            group_cols.append(c)

    rows: list[dict[str, Any]] = []
    for c in group_cols:
        tmp = df.dropna(subset=[c]).copy()
        if tmp.empty:
            continue
        g = tmp.groupby(c, dropna=True)
        agg = g.agg(
            n=("abs_error", "size"),
            mae=("abs_error", "mean"),
            rmse=("sq_error", lambda x: float(np.sqrt(np.nanmean(x)))),
            med_abs_error=("abs_error", "median"),
            p90_abs_error=("abs_error", lambda x: float(np.nanquantile(x, 0.9))),
            large_error_rate=("is_large_error", "mean"),
        ).reset_index()
        agg = agg.rename(columns={c: "bucket"})
        agg.insert(0, "dimension", c)
        rows.append(agg)

    if not rows:
        return pd.DataFrame(columns=["dimension", "bucket", "n", "mae", "rmse", "med_abs_error", "p90_abs_error", "large_error_rate"])
    return pd.concat(rows, axis=0, ignore_index=True)


def main() -> int:
    args = _parse_args()
    try:
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore
    except Exception as e:
        print(f"fatal: missing dependency numpy/pandas: {e}")
        return 2

    cme_csv = Path(args.cme_csv)
    if not cme_csv.exists():
        print(f"fatal: missing cme csv: {cme_csv}")
        return 2

    panel_csv = Path(args.panel_csv)
    sentiment_csv = Path(args.sentiment_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cme_csv)
    if df.empty:
        print("fatal: cme csv is empty")
        return 2
    rows_input = int(df.shape[0])

    date_col = _build_date_col(df)
    df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"]).copy()

    market_price_col = str(args.price_col).strip() or _pick_col(
        list(df.columns),
        ("market_price", "trade_price", "traded_price", "last_price", "price", "settle", "settlement_price"),
    )
    if not market_price_col:
        print("fatal: cannot find market traded price column; pass --price-col")
        return 2
    df["market_price"] = pd.to_numeric(df[market_price_col], errors="coerce")

    bs_price_col = str(args.bs_col).strip() or _pick_col(
        list(df.columns),
        ("bs_price", "bsm_price", "baseline_price", "model_price"),
    )

    if bs_price_col:
        df["bs_price"] = pd.to_numeric(df[bs_price_col], errors="coerce")
    else:
        spot_col = _pick_col(
            list(df.columns),
            ("S0", "s0", "spot", "spot_price", "underlying_price", "equity_close", "close"),
        )
        strike_col = _pick_col(
            list(df.columns),
            ("K", "k", "strike", "strike_price", "option_strike_price"),
        )
        t_col = _pick_col(
            list(df.columns),
            ("T_years", "t_years", "tau", "ttm", "time_to_maturity", "t"),
        )
        t_days_col = _pick_col(list(df.columns), ("days_to_expiry", "days_to_maturity", "dte"))
        r_col = _pick_col(list(df.columns), ("r", "risk_free_rate", "rate"))
        q_col = _pick_col(list(df.columns), ("q", "dividend_yield", "div_yield"))
        sigma_col = _pick_col(
            list(df.columns),
            ("sigma", "iv", "implied_vol", "implied_volatility", "option_implied_volatility"),
        )
        right_col = _pick_col(list(df.columns), ("option_type", "right", "type", "call_put", "option_side"))

        if spot_col is None or strike_col is None or sigma_col is None or (t_col is None and t_days_col is None):
            print(
                "fatal: cannot auto-compute bs_price. Need columns for spot/strike/sigma/T "
                "(or provide --bs-col with precomputed BSM price)."
            )
            return 2

        def _calc_row(row: Any) -> float | None:
            s = _safe_float(row.get(spot_col))
            k = _safe_float(row.get(strike_col))
            sigma = _normalize_sigma(row.get(sigma_col))
            if s is None or k is None or s <= 0 or k <= 0:
                return None
            if t_col is not None:
                t = _safe_float(row.get(t_col))
            else:
                dt = _safe_float(row.get(t_days_col))
                t = (float(dt) / 365.0) if dt is not None else None
            if t is None:
                return None
            t = max(float(t), 0.0)
            r = _normalize_rate(row.get(r_col)) if r_col is not None else 0.0
            q = _normalize_rate(row.get(q_col)) if q_col is not None else 0.0
            is_call = _bool_is_call(row.get(right_col)) if right_col is not None else True
            try:
                return float(_bs_price(is_call=is_call, s=float(s), k=float(k), t=t, r=r, q=q, sigma=sigma))
            except Exception:
                return None

        df["bs_price"] = df.apply(_calc_row, axis=1)
        bs_price_col = "computed_bs_price"

    # Optional joins: panel + sentiment for regime tagging.
    if panel_csv.exists():
        panel = pd.read_csv(panel_csv)
        if not panel.empty:
            panel_date_col = _pick_col(list(panel.columns), ("date", "trade_date", "datetime", "timestamp"))
            if panel_date_col:
                panel["date"] = pd.to_datetime(panel[panel_date_col], errors="coerce").dt.normalize()
                keep_panel_cols = ["date"]
                for c in ("vol_close", "equity_rvol_21", "equity_rvol_63", "equity_rvol_252"):
                    if c in panel.columns:
                        keep_panel_cols.append(c)
                panel = panel[keep_panel_cols].dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")
                df = df.merge(panel, on="date", how="left")

    if sentiment_csv.exists():
        s = pd.read_csv(sentiment_csv)
        if not s.empty and "date" in s.columns:
            score_col = _pick_col(list(s.columns), ("news_sent_01", "sentiment", "sentiment_score", "score"))
            if score_col:
                s["date"] = pd.to_datetime(s["date"], errors="coerce").dt.normalize()
                s["news_sent_01"] = pd.to_numeric(s[score_col], errors="coerce")
                s = s[["date", "news_sent_01"]].dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")
                df = df.merge(s, on="date", how="left")

    # Build regime proxies.
    vol_proxy_col = _pick_col(
        list(df.columns),
        (
            "equity_rvol_21",
            "equity_rvol_63",
            "vol_close",
            "sigma",
            "iv",
            "implied_volatility",
            "option_implied_volatility",
        ),
    )
    if vol_proxy_col:
        df["vol_proxy"] = pd.to_numeric(df[vol_proxy_col], errors="coerce")
        if vol_proxy_col == "vol_close":
            df["vol_proxy"] = df["vol_proxy"] / 100.0
        if df["vol_proxy"].dropna().size > 0:
            vol_q = float(np.clip(float(args.vol_quantile), 0.0, 1.0))
            vol_threshold = float(df["vol_proxy"].quantile(vol_q))
            df["is_high_vol"] = df["vol_proxy"] >= vol_threshold
        else:
            vol_threshold = None
            df["is_high_vol"] = False
    else:
        vol_threshold = None
        df["is_high_vol"] = False

    if "news_sent_01" in df.columns:
        sent_by_day = (
            df[["date", "news_sent_01"]]
            .dropna(subset=["date"])
            .drop_duplicates(subset=["date"], keep="last")
            .sort_values("date")
            .copy()
        )
        sent_by_day["sent_delta_abs"] = sent_by_day["news_sent_01"].diff().abs()
        df = df.merge(sent_by_day[["date", "sent_delta_abs"]], on="date", how="left")
        if sent_by_day["sent_delta_abs"].dropna().size > 0:
            sent_q = float(np.clip(float(args.sent_delta_quantile), 0.0, 1.0))
            sent_delta_threshold = float(sent_by_day["sent_delta_abs"].quantile(sent_q))
            df["is_sent_shift"] = df["sent_delta_abs"] >= sent_delta_threshold
        else:
            sent_delta_threshold = None
            df["is_sent_shift"] = False
    else:
        sent_delta_threshold = None
        df["sent_delta_abs"] = np.nan
        df["is_sent_shift"] = False

    eval_df = df.dropna(subset=["market_price", "bs_price"]).copy()
    if "volume" in eval_df.columns and float(args.min_volume) > 0:
        eval_df["volume"] = pd.to_numeric(eval_df["volume"], errors="coerce")
        eval_df = eval_df[eval_df["volume"] >= float(args.min_volume)]
    if "open_interest" in eval_df.columns and float(args.min_open_interest) > 0:
        eval_df["open_interest"] = pd.to_numeric(eval_df["open_interest"], errors="coerce")
        eval_df = eval_df[eval_df["open_interest"] >= float(args.min_open_interest)]
    if float(args.min_market_price) > 0:
        eval_df = eval_df[eval_df["market_price"] >= float(args.min_market_price)]
    if eval_df.empty:
        print("fatal: no valid rows after filtering market_price and bs_price")
        return 2

    eval_df["error"] = eval_df["bs_price"] - eval_df["market_price"]
    eval_df["abs_error"] = eval_df["error"].abs()
    eval_df["sq_error"] = eval_df["error"] ** 2
    eval_df["is_high_vol"] = eval_df["is_high_vol"].fillna(False).astype(bool)
    eval_df["is_sent_shift"] = eval_df["is_sent_shift"].fillna(False).astype(bool)

    overall = _compute_metrics(eval_df, "overall")
    by_regime = [
        _compute_metrics(eval_df[eval_df["is_high_vol"]], "high_vol"),
        _compute_metrics(eval_df[~eval_df["is_high_vol"]], "non_high_vol"),
        _compute_metrics(eval_df[eval_df["is_sent_shift"]], "sent_shift"),
        _compute_metrics(eval_df[~eval_df["is_sent_shift"]], "sent_stable"),
        _compute_metrics(eval_df[eval_df["is_high_vol"] & eval_df["is_sent_shift"]], "high_vol_and_sent_shift"),
        _compute_metrics(eval_df[~eval_df["is_high_vol"] & ~eval_df["is_sent_shift"]], "normal_regime"),
    ]

    result = EvalResult(
        rows_input=rows_input,
        rows_eval=int(eval_df.shape[0]),
        market_price_col=market_price_col,
        bs_price_col=str(bs_price_col),
        vol_proxy_col=vol_proxy_col,
        vol_threshold=vol_threshold,
        sent_delta_threshold=sent_delta_threshold,
    )

    eval_out = out_dir / "evaluated_predictions.csv"
    overall_out = out_dir / "metrics_overall.csv"
    regime_out = out_dir / "metrics_by_regime.csv"
    profile_out = out_dir / "metrics_error_profile.csv"
    summary_out = out_dir / "summary.json"

    eval_df.sort_values(["date"]).to_csv(eval_out, index=False)
    pd.DataFrame([overall]).to_csv(overall_out, index=False)
    pd.DataFrame(by_regime).to_csv(regime_out, index=False)
    _build_error_profile(eval_df).to_csv(profile_out, index=False)
    summary_out.write_text(
        json.dumps(
            {
                "ok": True,
                "inputs": {
                    "cme_csv": str(cme_csv),
                    "panel_csv": str(panel_csv) if panel_csv.exists() else None,
                    "sentiment_csv": str(sentiment_csv) if sentiment_csv.exists() else None,
                },
                "result": result.__dict__,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"saved: {eval_out}")
    print(f"saved: {overall_out}")
    print(f"saved: {regime_out}")
    print(f"saved: {profile_out}")
    print(f"saved: {summary_out}")
    print(f"overall MAE={overall['mae']:.6f} RMSE={overall['rmse']:.6f} rows={overall['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
