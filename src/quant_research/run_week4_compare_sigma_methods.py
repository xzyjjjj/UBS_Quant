#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare Week4 baseline accuracy under different sigma constructions."
    )
    p.add_argument(
        "--eval-csv",
        default="output/cme/jpm_options.csv",
        help="Prepared eval input CSV (default: output/cme/jpm_options.csv)",
    )
    p.add_argument(
        "--panel-csv",
        default="output/market_panel/processed/panel.csv",
        help="Panel CSV containing equity_rvol_* (default: output/market_panel/processed/panel.csv)",
    )
    p.add_argument(
        "--out-csv",
        default="output/week4_baseline/sigma_method_comparison.csv",
        help="Output comparison CSV (default: output/week4_baseline/sigma_method_comparison.csv)",
    )
    p.add_argument(
        "--min-volume",
        type=float,
        default=0.0,
        help="Optional liquidity filter: volume >= min-volume (default: 0)",
    )
    p.add_argument(
        "--min-open-interest",
        type=float,
        default=0.0,
        help="Optional liquidity filter: open_interest >= min-open-interest (default: 0)",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Print top-k methods sorted by MAE/RMSE (default: 20)",
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


def _metrics(df: Any, sigma_col: str, label: str) -> dict[str, Any]:
    import numpy as np

    tmp = df.dropna(subset=["market_price", "S0", "strike_price", "T_years", "r", "q", sigma_col]).copy()
    if tmp.empty:
        return {"method": label, "n": 0, "mae": None, "rmse": None}

    sig = tmp[sigma_col].to_numpy(dtype=float)
    s0 = tmp["S0"].to_numpy(dtype=float)
    k = tmp["strike_price"].to_numpy(dtype=float)
    t = tmp["T_years"].to_numpy(dtype=float)
    r = tmp["r"].to_numpy(dtype=float)
    q = tmp["q"].to_numpy(dtype=float)
    mkt = tmp["market_price"].to_numpy(dtype=float)
    is_call = tmp["option_type"].astype(str).str.lower().str.contains("c").to_numpy()

    pred = np.empty(len(tmp), dtype=float)
    for i in range(len(tmp)):
        pred[i] = _bs_price(
            is_call=bool(is_call[i]),
            s=float(s0[i]),
            k=float(k[i]),
            t=max(0.0, float(t[i])),
            r=float(r[i]),
            q=float(q[i]),
            sigma=max(1e-8, float(sig[i])),
        )
    err = pred - mkt
    return {
        "method": label,
        "n": int(len(tmp)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err * err))),
    }


def main() -> int:
    args = _parse_args()
    try:
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore
    except Exception as e:
        print(f"fatal: missing dependency numpy/pandas: {e}")
        return 2

    eval_csv = Path(args.eval_csv)
    panel_csv = Path(args.panel_csv)
    out_csv = Path(args.out_csv)
    if not eval_csv.exists():
        print(f"fatal: missing eval csv: {eval_csv}")
        return 2
    if not panel_csv.exists():
        print(f"fatal: missing panel csv: {panel_csv}")
        return 2

    df = pd.read_csv(eval_csv)
    panel = pd.read_csv(panel_csv)
    if df.empty:
        print("fatal: eval csv is empty")
        return 2

    # Normalize and merge rv columns.
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    keep = ["date"] + [
        c
        for c in (
            "equity_close",
            "equity_adj_close",
            "equity_rvol_21",
            "equity_rvol_63",
            "equity_rvol_252",
        )
        if c in panel.columns
    ]
    panel = panel[keep].drop_duplicates(subset=["date"], keep="last")

    # If panel has equity price, compute additional RV windows for richer sigma search.
    px_col = "equity_close" if "equity_close" in panel.columns else ("equity_adj_close" if "equity_adj_close" in panel.columns else None)
    if px_col is not None:
        px = pd.to_numeric(panel[px_col], errors="coerce")
        ret = np.log(px.where(px > 0)).diff()
        for w in (126, 252):
            col = f"equity_rvol_{w}"
            if col not in panel.columns:
                panel[col] = ret.rolling(window=w, min_periods=w).std(ddof=1) * np.sqrt(252.0)

    df = df.merge(panel, on="date", how="left")

    # Numeric coercion
    for c in ("market_price", "S0", "strike_price", "T_years", "r", "q", "sigma", "volume", "open_interest"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    rvol_cols = [c for c in df.columns if c.startswith("equity_rvol_")]
    for c in rvol_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Optional liquidity filters
    base = df.copy()
    if float(args.min_volume) > 0 and "volume" in base.columns:
        base = base[base["volume"] >= float(args.min_volume)]
    if float(args.min_open_interest) > 0 and "open_interest" in base.columns:
        base = base[base["open_interest"] >= float(args.min_open_interest)]

    # Build sigma variants
    base["sig_iv"] = base["sigma"]
    for c in sorted([c for c in base.columns if c.startswith("equity_rvol_")]):
        w = c.split("_")[-1]
        base[f"sig_rv{w}"] = base[c]

    # Winsorized IV
    iv = base["sigma"]
    if iv.dropna().size > 0:
        lo = float(iv.quantile(0.01))
        hi = float(iv.quantile(0.99))
        base["sig_iv_winsor_1_99"] = iv.clip(lower=lo, upper=hi)

    # Build blend candidates: iv with each rv window under multiple weights.
    rv_sig_cols = sorted(
        [c for c in base.columns if c in ("sig_rv126", "sig_rv252")],
        key=lambda x: int(x.replace("sig_rv", "")),
    )
    for rv_col in rv_sig_cols:
        rv_label = rv_col.replace("sig_", "")
        for w_iv in (0.25, 0.5, 0.75):
            w_rv = 1.0 - w_iv
            blend_col = f"sig_blend_iv_{rv_label}_{int(w_iv*100):02d}_{int(w_rv*100):02d}"
            base[blend_col] = w_iv * base["sigma"] + w_rv * base[rv_col]
        # Harmonic mean blend can be robust to extremes.
        harm_col = f"sig_hmean_iv_{rv_label}"
        base[harm_col] = 2.0 / ((1.0 / base["sigma"]) + (1.0 / base[rv_col]))

    methods = []
    # Base IV methods
    methods.append(_metrics(base, "sig_iv", "iv"))
    if "sig_iv_winsor_1_99" in base.columns:
        methods.append(_metrics(base, "sig_iv_winsor_1_99", "iv_winsor_1_99"))
    # RV + blend methods
    for c in sorted([c for c in base.columns if c.startswith("sig_rv") or c.startswith("sig_blend_iv_") or c.startswith("sig_hmean_iv_")]):
        methods.append(_metrics(base, c, c.replace("sig_", "")))

    out = pd.DataFrame(methods).sort_values(["mae", "rmse"], na_position="last")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"saved: {out_csv}")
    top_k = max(1, int(args.top_k))
    print(out.head(top_k).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
