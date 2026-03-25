#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Price vanilla options using BSM (theoretical) and optionally solve IV from market quotes."
    )
    p.add_argument(
        "--merged-csv",
        default="output/futu/option_chain_quotes_HK_00700_all.csv",
        help="CSV with option chain merged with option quotes (default: output/futu/option_chain_quotes_HK_00700_all.csv)",
    )
    p.add_argument(
        "--snapshot-csv",
        default="output/futu/snapshot.csv",
        help="Underlying snapshot CSV containing last_price (default: output/futu/snapshot.csv)",
    )
    p.add_argument(
        "--out-csv",
        default="output/futu/vanilla_pricing.csv",
        help="Output pricing CSV path (default: output/futu/vanilla_pricing.csv)",
    )
    p.add_argument(
        "--right",
        default="ALL",
        choices=("ALL", "CALL", "PUT"),
        help="Filter rows by CALL/PUT when possible (default: ALL)",
    )
    p.add_argument(
        "--expiry",
        default="",
        help="Filter rows by expiry date (YYYY-MM-DD) when possible (default: no filter)",
    )
    p.add_argument("--r", type=float, default=float(os.getenv("FUTU_R", "0") or 0), help="Risk-free rate (cont comp)")
    p.add_argument("--q", type=float, default=float(os.getenv("FUTU_Q", "0") or 0), help="Dividend yield (cont comp)")
    p.add_argument(
        "--sigma",
        type=float,
        default=float(os.getenv("FUTU_SIGMA", "0") or 0),
        help="Vol used for BSM theoretical pricing (annualized). Required unless you only want IV solving.",
    )
    p.add_argument(
        "--use-mid",
        action="store_true",
        help="Use mid price (bid/ask) when available for IV solving (default: off)",
    )
    p.add_argument(
        "--solve-iv",
        action="store_true",
        help="Solve implied vol per contract using market price (default: off)",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=500,
        help="Max rows to price/solve (default: 500)",
    )
    return p.parse_args()


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        v = float(x)
        return v if math.isfinite(v) else None
    s = str(x).strip()
    if not s or s.upper() == "N/A":
        return None
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _parse_yyyy_mm_dd(s: str) -> date | None:
    s2 = (s or "").strip()
    if not s2:
        return None
    try:
        return datetime.strptime(s2, "%Y-%m-%d").date()
    except Exception:
        return None


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_vanilla_price(*, is_call: bool, s: float, k: float, t: float, r: float, q: float, sigma: float) -> float:
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


def _implied_vol_bisect(
    *,
    is_call: bool,
    s: float,
    k: float,
    t: float,
    r: float,
    q: float,
    price: float,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> float | None:
    if t <= 0 or price <= 0:
        return None
    df_r = math.exp(-r * t)
    df_q = math.exp(-q * t)
    lower = max(0.0, s * df_q - k * df_r) if is_call else max(0.0, k * df_r - s * df_q)
    upper = s * df_q if is_call else k * df_r
    if price < lower - 1e-8 or price > upper + 1e-8:
        return None

    lo, hi = 1e-6, 5.0
    plo = _bs_vanilla_price(is_call=is_call, s=s, k=k, t=t, r=r, q=q, sigma=lo)
    phi = _bs_vanilla_price(is_call=is_call, s=s, k=k, t=t, r=r, q=q, sigma=hi)
    while phi < price and hi < 20.0:
        hi *= 2.0
        phi = _bs_vanilla_price(is_call=is_call, s=s, k=k, t=t, r=r, q=q, sigma=hi)
    if not (plo <= price <= phi):
        return None

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        pmid = _bs_vanilla_price(is_call=is_call, s=s, k=k, t=t, r=r, q=q, sigma=mid)
        if abs(pmid - price) <= tol:
            return mid
        if pmid < price:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class PricingSummary:
    ok: bool
    rows: int
    out_csv: str
    s0: float
    sigma: float | None
    solve_iv: bool


def main() -> int:
    args = _parse_args()

    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        print(f"fatal: missing dependency pandas: {e}")
        return 2

    merged_csv = Path(args.merged_csv)
    snapshot_csv = Path(args.snapshot_csv)
    out_csv = Path(args.out_csv)

    if not snapshot_csv.exists():
        print(f"fatal: missing snapshot csv: {snapshot_csv}")
        return 2
    snap = pd.read_csv(snapshot_csv)
    if snap.empty or "last_price" not in snap.columns:
        print("fatal: snapshot csv missing last_price")
        return 2
    s0 = _safe_float(snap.loc[0, "last_price"])
    if s0 is None:
        print("fatal: snapshot last_price invalid")
        return 2

    if not merged_csv.exists():
        print(f"fatal: missing merged csv: {merged_csv}")
        print("hint: run `run_option_get_data.py --save-merged` first.")
        return 2
    df = pd.read_csv(merged_csv)
    if df.empty:
        print("fatal: merged csv is empty")
        return 2

    # Filters (best-effort)
    if str(args.expiry).strip():
        for col in ("strike_time", "expiry_date", "maturity_date"):
            if col in df.columns:
                df = df[df[col].astype(str) == str(args.expiry).strip()]
                break
    if str(args.right).upper() in ("CALL", "PUT"):
        for col in ("option_type", "type", "call_put", "option_side"):
            if col in df.columns:
                df = df[df[col].astype(str).str.upper().str.contains(str(args.right).upper())]
                break

    df = df.head(int(args.max_rows)).copy()

    strike_col = "strike_price" if "strike_price" in df.columns else None
    expiry_col = None
    for c in ("strike_time", "expiry_date", "maturity_date"):
        if c in df.columns:
            expiry_col = c
            break
    right_col = None
    for c in ("option_type", "type", "call_put", "option_side"):
        if c in df.columns:
            right_col = c
            break

    if strike_col is None or expiry_col is None:
        print("fatal: merged csv missing strike_price or expiry column (strike_time/expiry_date)")
        return 2

    # Market price (best-effort)
    bid_col = "bid_price" if "bid_price" in df.columns else None
    ask_col = "ask_price" if "ask_price" in df.columns else None
    last_col = "last_price" if "last_price" in df.columns else None

    r = float(args.r)
    q = float(args.q)
    sigma = float(args.sigma or 0.0)
    solve_iv = bool(args.solve_iv)
    if sigma <= 0 and not solve_iv:
        print("fatal: --sigma is required for BSM theoretical pricing (or pass --solve-iv only).")
        return 2

    t_list: list[float | None] = []
    mkt_px_list: list[float | None] = []
    bs_list: list[float | None] = []
    iv_list: list[float | None] = []

    today = date.today()
    for _, row in df.iterrows():
        k_val = _safe_float(row.get(strike_col))
        exp_dt = _parse_yyyy_mm_dd(str(row.get(expiry_col)))
        t = max(0.0, (exp_dt - today).days / 365.0) if exp_dt is not None else None
        t_list.append(t)

        bid = _safe_float(row.get(bid_col)) if bid_col else None
        ask = _safe_float(row.get(ask_col)) if ask_col else None
        last = _safe_float(row.get(last_col)) if last_col else None
        mid = 0.5 * (bid + ask) if (bid is not None and ask is not None and ask >= bid) else None
        mkt_px = mid if bool(args.use_mid) and mid is not None else (last or mid)
        mkt_px_list.append(mkt_px)

        is_call = True
        if right_col:
            is_call = "CALL" in str(row.get(right_col)).upper()

        if k_val is None or t is None:
            bs_list.append(None)
            iv_list.append(None)
            continue

        if sigma > 0:
            bs_list.append(
                _bs_vanilla_price(
                    is_call=is_call,
                    s=float(s0),
                    k=float(k_val),
                    t=float(t),
                    r=float(r),
                    q=float(q),
                    sigma=float(sigma),
                )
            )
        else:
            bs_list.append(None)

        if solve_iv and mkt_px is not None and mkt_px > 0 and t > 0:
            iv_list.append(
                _implied_vol_bisect(
                    is_call=is_call,
                    s=float(s0),
                    k=float(k_val),
                    t=float(t),
                    r=float(r),
                    q=float(q),
                    price=float(mkt_px),
                )
            )
        else:
            iv_list.append(None)

    df = df.assign(S0=float(s0), r=float(r), q=float(q), T_years=t_list, market_price=mkt_px_list, bs_price=bs_list, iv=iv_list)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"saved: {out_csv}")

    summary = PricingSummary(
        ok=True,
        rows=int(getattr(df, "shape", [0])[0]),
        out_csv=str(out_csv),
        s0=float(s0),
        sigma=float(sigma) if sigma > 0 else None,
        solve_iv=solve_iv,
    )
    meta_path = out_csv.with_suffix(".json")
    meta_path.write_text(json.dumps(summary.__dict__, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved: {meta_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

