#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Price a simple chooser option under BSM via Monte Carlo (option chosen at T1, expires at T2)."
    )
    p.add_argument(
        "--snapshot-csv",
        default="output/futu/snapshot.csv",
        help="CSV containing market snapshot with last_price (default: output/futu/snapshot.csv)",
    )
    p.add_argument(
        "--kline-csv",
        default="output/futu/kline_HK_00700.csv",
        help="CSV containing historical kline with close/time_key (default: output/futu/kline_HK_00700.csv)",
    )
    p.add_argument("--s0", type=float, default=0.0, help="Spot price override (default: read from snapshot)")
    p.add_argument("--k", type=float, default=150.0, help="Strike K (default: 150)")
    p.add_argument("--t1", type=float, default=0.5, help="Chooser decision time in years (default: 0.5)")
    p.add_argument("--t2", type=float, default=1.0, help="Option maturity in years (default: 1.0)")
    p.add_argument("--r", type=float, default=0.0, help="Risk-free rate (cont comp, annualized; default: 0)")
    p.add_argument("--q", type=float, default=0.0, help="Dividend yield (cont comp, annualized; default: 0)")
    p.add_argument("--sigma", type=float, default=0.0, help="Vol override (annualized; default: compute from kline)")
    p.add_argument("--sigma-window", type=int, default=252, help="Returns window for hist vol (default: 252)")
    p.add_argument("--trading-days", type=int, default=252, help="Trading days per year (default: 252)")
    p.add_argument("--n-paths", type=int, default=200_000, help="MC paths (default: 200000)")
    p.add_argument("--seed", type=int, default=7, help="RNG seed (default: 7)")
    p.add_argument(
        "--out-json",
        default="output/futu/chooser_mc.json",
        help="Write run summary as JSON (default: output/futu/chooser_mc.json)",
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


def _read_s0(snapshot_csv: Path) -> float:
    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Missing dependency: pandas ({e}). Provide --s0 to bypass reading snapshot.") from e

    if not snapshot_csv.exists():
        raise FileNotFoundError(f"snapshot csv not found: {snapshot_csv}")
    df = pd.read_csv(snapshot_csv)
    if df.empty:
        raise RuntimeError("snapshot csv is empty")
    v = _safe_float(df.loc[0, "last_price"]) if "last_price" in df.columns else None
    if v is None:
        raise RuntimeError("snapshot csv missing last_price or value is invalid")
    return float(v)


def _compute_hist_sigma(kline_csv: Path, *, window: int, trading_days: int) -> float:
    try:
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore
    except Exception as e:
        raise RuntimeError(
            f"Missing dependency: numpy/pandas ({e}). Provide --sigma to bypass hist vol."
        ) from e

    if not kline_csv.exists():
        raise FileNotFoundError(f"kline csv not found: {kline_csv}")
    df = pd.read_csv(kline_csv)
    if df.empty or "close" not in df.columns:
        raise RuntimeError("kline csv is empty or missing close column")
    if "time_key" in df.columns:
        try:
            df = df.sort_values("time_key")
        except Exception:
            pass

    px = pd.to_numeric(df["close"], errors="coerce").dropna()
    if len(px) < 3:
        raise RuntimeError("not enough close prices to compute volatility")
    rets = np.log(px).diff().dropna()
    if window and window > 0:
        rets = rets.tail(int(window))
    if len(rets) < 2:
        raise RuntimeError("not enough returns after windowing")

    sigma_daily = float(rets.std(ddof=1))
    return sigma_daily * math.sqrt(float(trading_days))


@dataclass(frozen=True)
class MCResult:
    price: float
    stderr: float
    ci95_low: float
    ci95_high: float
    p_choose_call: float


def _mc_chooser_price_numpy(
    *,
    s0: float,
    k: float,
    t1: float,
    t2: float,
    r: float,
    q: float,
    sigma: float,
    n_paths: int,
    seed: int,
) -> MCResult:
    import numpy as np  # type: ignore

    if not (0.0 < t1 < t2):
        raise ValueError("Require 0 < T1 < T2")
    if sigma <= 0:
        raise ValueError("sigma must be > 0")

    rng = np.random.default_rng(int(seed))

    dt1 = float(t1)
    dt2 = float(t2 - t1)
    drift1 = (r - q - 0.5 * sigma * sigma) * dt1
    drift2 = (r - q - 0.5 * sigma * sigma) * dt2
    vol1 = sigma * math.sqrt(dt1)
    vol2 = sigma * math.sqrt(dt2)

    z1 = rng.standard_normal(int(n_paths))
    z2 = rng.standard_normal(int(n_paths))
    s_t1 = s0 * np.exp(drift1 + vol1 * z1)
    s_t2 = s_t1 * np.exp(drift2 + vol2 * z2)

    s_star = k * math.exp(-(r - q) * (t2 - t1))
    choose_call = s_t1 >= s_star
    payoff = np.where(choose_call, np.maximum(s_t2 - k, 0.0), np.maximum(k - s_t2, 0.0))

    disc = math.exp(-r * t2)
    pv = disc * payoff
    price = float(pv.mean())
    stderr = float(pv.std(ddof=1) / math.sqrt(float(n_paths)))
    ci95 = 1.96 * stderr
    return MCResult(
        price=price,
        stderr=stderr,
        ci95_low=price - ci95,
        ci95_high=price + ci95,
        p_choose_call=float(choose_call.mean()),
    )


def _mc_chooser_price_purepy(
    *,
    s0: float,
    k: float,
    t1: float,
    t2: float,
    r: float,
    q: float,
    sigma: float,
    n_paths: int,
    seed: int,
) -> MCResult:
    import random

    if not (0.0 < t1 < t2):
        raise ValueError("Require 0 < T1 < T2")
    if sigma <= 0:
        raise ValueError("sigma must be > 0")

    random.seed(int(seed))
    dt1 = float(t1)
    dt2 = float(t2 - t1)
    drift1 = (r - q - 0.5 * sigma * sigma) * dt1
    drift2 = (r - q - 0.5 * sigma * sigma) * dt2
    vol1 = sigma * math.sqrt(dt1)
    vol2 = sigma * math.sqrt(dt2)
    disc = math.exp(-r * t2)
    s_star = k * math.exp(-(r - q) * (t2 - t1))

    s1_list: list[float] = []
    pv_list: list[float] = []
    choose_call_count = 0

    for _ in range(int(n_paths)):
        # Box-Muller for normals
        u1 = max(1e-12, random.random())
        u2 = random.random()
        z1 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        u3 = max(1e-12, random.random())
        u4 = random.random()
        z2 = math.sqrt(-2.0 * math.log(u3)) * math.cos(2.0 * math.pi * u4)

        s_t1 = s0 * math.exp(drift1 + vol1 * z1)
        s_t2 = s_t1 * math.exp(drift2 + vol2 * z2)
        is_call = s_t1 >= s_star
        choose_call_count += 1 if is_call else 0

        payoff = max(s_t2 - k, 0.0) if is_call else max(k - s_t2, 0.0)
        pv_list.append(disc * payoff)
        s1_list.append(s_t1)

    n = float(n_paths)
    mean = sum(pv_list) / n
    var = sum((x - mean) ** 2 for x in pv_list) / (n - 1.0) if n_paths > 1 else 0.0
    stderr = math.sqrt(var) / math.sqrt(n)
    ci95 = 1.96 * stderr
    return MCResult(
        price=float(mean),
        stderr=float(stderr),
        ci95_low=float(mean - ci95),
        ci95_high=float(mean + ci95),
        p_choose_call=float(choose_call_count / n),
    )


def main() -> int:
    args = _parse_args()
    snapshot_csv = Path(args.snapshot_csv)
    kline_csv = Path(args.kline_csv)

    s0 = float(args.s0 or 0.0)
    if s0 <= 0:
        s0 = _read_s0(snapshot_csv)

    sigma = float(args.sigma or 0.0)
    if sigma <= 0:
        sigma = _compute_hist_sigma(
            kline_csv,
            window=int(args.sigma_window),
            trading_days=int(args.trading_days),
        )

    params = {
        "S0": float(s0),
        "K": float(args.k),
        "T1": float(args.t1),
        "T2": float(args.t2),
        "r": float(args.r),
        "q": float(args.q),
        "sigma": float(sigma),
        "n_paths": int(args.n_paths),
        "seed": int(args.seed),
        "snapshot_csv": str(snapshot_csv),
        "kline_csv": str(kline_csv),
        "asof": date.today().isoformat(),
    }

    try:
        try:
            import numpy as np  # type: ignore  # noqa: F401

            res = _mc_chooser_price_numpy(
                s0=float(s0),
                k=float(args.k),
                t1=float(args.t1),
                t2=float(args.t2),
                r=float(args.r),
                q=float(args.q),
                sigma=float(sigma),
                n_paths=int(args.n_paths),
                seed=int(args.seed),
            )
            engine = "numpy"
        except Exception:
            res = _mc_chooser_price_purepy(
                s0=float(s0),
                k=float(args.k),
                t1=float(args.t1),
                t2=float(args.t2),
                r=float(args.r),
                q=float(args.q),
                sigma=float(sigma),
                n_paths=int(args.n_paths),
                seed=int(args.seed),
            )
            engine = "purepy"
    except Exception as e:
        print(f"fatal: {e}")
        return 2

    print("\n== chooser_bsm_mc ==")
    print(f"engine={engine}")
    print(f"price={res.price:.6f}")
    print(f"stderr={res.stderr:.6f}")
    print(f"ci95=[{res.ci95_low:.6f}, {res.ci95_high:.6f}]")
    print(f"p_choose_call={res.p_choose_call:.4f}")
    print(
        f"inputs: S0={params['S0']:.6f} K={params['K']:.6f} "
        f"T1={params['T1']:.6f} T2={params['T2']:.6f} r={params['r']:.6f} q={params['q']:.6f} "
        f"sigma={params['sigma']:.6f} n={params['n_paths']} seed={params['seed']}"
    )

    out = {
        "ok": True,
        "engine": engine,
        "result": {
            "price": res.price,
            "stderr": res.stderr,
            "ci95_low": res.ci95_low,
            "ci95_high": res.ci95_high,
            "p_choose_call": res.p_choose_call,
        },
        "params": params,
        "notes": [
            "Chooser definition: at T1 choose the higher-value (call/put) with strike K and maturity T2.",
            "Decision boundary under BSM: choose call iff S(T1) >= K * exp(-(r-q)*(T2-T1)).",
        ],
    }

    try:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"saved: {out_path}")
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

