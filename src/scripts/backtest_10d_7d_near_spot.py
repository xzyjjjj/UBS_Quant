#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any


def safe_float(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    return v if math.isfinite(v) else None


def parse_date(s: str) -> date | None:
    s2 = str(s or "").strip()
    if not s2:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(s2[:10], fmt).date()
        except Exception:
            pass
    return None


def norm_type(s: str) -> str | None:
    x = str(s or "").strip().lower()
    if "call" in x or x in {"c", "1"}:
        return "call"
    if "put" in x or x in {"p", "2"}:
        return "put"
    return None


def norm_rate(v: float | None) -> float:
    if v is None:
        return 0.0
    return v / 100.0 if abs(v) > 3.0 else v


def norm_sigma(v: float | None) -> float:
    if v is None:
        return 0.0
    return v / 100.0 if v > 3.0 else v


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(*, is_call: bool, s: float, k: float, t: float, r: float, q: float, sigma: float) -> float:
    if t <= 0.0:
        return max(0.0, (s - k) if is_call else (k - s))
    sigma = max(sigma, 1e-12)
    sqrt_t = math.sqrt(max(t, 1e-12))
    d1 = (math.log(max(s, 1e-12) / max(k, 1e-12)) + (r - q + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    df_r = math.exp(-r * t)
    df_q = math.exp(-q * t)
    if is_call:
        return s * df_q * _norm_cdf(d1) - k * df_r * _norm_cdf(d2)
    return k * df_r * _norm_cdf(-d2) - s * df_q * _norm_cdf(-d1)


@dataclass
class Candidate:
    trade_date: date
    expiration: date
    option_type: str
    strike: float
    premium: float
    s0: float
    volume: float
    open_interest: float
    contract_id: str
    dte: int
    model_fair: float | None = None
    model_edge: float | None = None


def choose_one(cands: list[Candidate], spot: float, *, use_model_rank: bool) -> Candidate:
    if use_model_rank:
        return sorted(
            cands,
            key=lambda c: (
                -(c.model_edge if c.model_edge is not None else -1e18),
                abs(c.strike - spot),
                -c.volume,
                c.premium,
                c.contract_id,
            ),
        )[0]
    return sorted(
        cands,
        key=lambda c: (
            abs(c.strike - spot),
            -c.volume,
            c.premium,
            c.contract_id,
        ),
    )[0]


def choose_one_relaxed(cands: list[Candidate], spot: float, target_dte: int, *, use_model_rank: bool) -> Candidate:
    if use_model_rank:
        return sorted(
            cands,
            key=lambda c: (
                -(c.model_edge if c.model_edge is not None else -1e18),
                abs(c.dte - target_dte),
                abs(c.strike - spot),
                -c.volume,
                c.premium,
                c.contract_id,
            ),
        )[0]
    return sorted(
        cands,
        key=lambda c: (
            abs(c.dte - target_dte),
            abs(c.strike - spot),
            -c.volume,
            c.premium,
            c.contract_id,
        ),
    )[0]


def payoff(opt_type: str, s_exp: float, strike: float) -> float:
    if opt_type == "call":
        return max(s_exp - strike, 0.0)
    return max(strike - s_exp, 0.0)


def nearest_settle_date(all_dates: list[date], target: date) -> date | None:
    if not all_dates:
        return None
    later = [d for d in all_dates if d >= target]
    if later:
        return later[0]
    return all_dates[-1]


def load_model(model_json: str) -> tuple[str | None, dict[str, Any] | None]:
    p = str(model_json or "").strip()
    if not p:
        return None, None
    d = json.loads(Path(p).read_text(encoding="utf-8"))
    name = str(d.get("model_name", ""))
    if name == "mlp_direct":
        m = d["model"]
    elif name == "bsm_mlp_residual":
        m = d["model"]["residual_model"]
    else:
        raise ValueError(f"Unsupported model_name={name}")
    model = {
        "mean": [float(x) for x in d["normalization"]["mean"]],
        "std": [float(x) for x in d["normalization"]["std"]],
        "w1": [[float(z) for z in row] for row in m["w1"]],
        "b1": [float(x) for x in m["b1"]],
        "w2": [float(x) for x in m["w2"]],
        "b2": float(m["b2"]),
    }
    return name, model


def mlp_predict(xn: list[float], model: dict[str, Any]) -> float:
    z = []
    for j, row in enumerate(model["w1"]):
        s = model["b1"][j]
        for a, b in zip(row, xn):
            s += a * b
        z.append(s if s > 0 else 0.0)
    y = model["b2"]
    for w, a in zip(model["w2"], z):
        y += w * a
    return y


def model_fair_price(*, model_kind: str | None, model: dict[str, Any] | None, s0: float, strike: float, dte: int, sigma: float, r: float, q: float, opt: str) -> float | None:
    if model_kind is None or model is None or dte <= 0:
        return None
    t = dte / 365.0
    mny = s0 / strike
    x = [
        math.log(max(mny, 1e-12)),
        t,
        sigma,
        r,
        q,
        1.0 if opt == "call" else 0.0,
        math.sqrt(max(t, 0.0)),
        mny,
    ]
    xn = [(x[i] - model["mean"][i]) / model["std"][i] for i in range(len(x))]
    if model_kind == "mlp_direct":
        return mlp_predict(xn, model)
    # BSM + residual
    base = bs_price(is_call=(opt == "call"), s=s0, k=strike, t=t, r=r, q=q, sigma=max(sigma, 1e-12))
    return base + mlp_predict(xn, model)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="Backtest: buy 1 option/day and hold to expiry.")
    p.add_argument("--data-csv", default=str(root / "output" / "server_bundle_week5" / "data" / "processed" / "jpm_options_final.csv"))
    p.add_argument("--symbol", default="JPM")
    p.add_argument("--option-type", default="call", choices=("call", "put", "both"))
    p.add_argument("--lookback-days", type=int, default=10)
    p.add_argument("--target-dte", type=int, default=7)
    p.add_argument("--dte-tol", type=int, default=1)
    p.add_argument("--max-strike-gap", type=float, default=5.0)
    p.add_argument("--contract-multiplier", type=float, default=100.0)
    p.add_argument("--force-daily-trade", action="store_true")
    p.add_argument("--model-json", default="")
    p.add_argument("--use-model-rank", action="store_true")
    p.add_argument("--min-model-edge", type=float, default=-1e18)
    p.add_argument("--min-volume", type=float, default=0.0)
    p.add_argument("--min-open-interest", type=float, default=0.0)
    p.add_argument("--max-moneyness-dev", type=float, default=0.20)
    p.add_argument("--out-csv", default=str(root / "output" / "week6_backtest_10d_7d_near_spot.csv"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    data_csv = Path(args.data_csv)
    model_kind, model = load_model(args.model_json)

    date_set: set[date] = set()
    spot_sum: dict[date, float] = {}
    spot_n: dict[date, int] = {}

    with data_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("symbol", "")).upper() != str(args.symbol).upper():
                continue
            d = parse_date(row.get("date", ""))
            s0 = safe_float(row.get("S0"))
            if d is None or s0 is None or s0 <= 0:
                continue
            date_set.add(d)
            spot_sum[d] = spot_sum.get(d, 0.0) + s0
            spot_n[d] = spot_n.get(d, 0) + 1

    all_dates = sorted(date_set)
    if not all_dates:
        raise RuntimeError("No valid dates found in dataset.")
    last_dates = all_dates[-int(args.lookback_days):]
    last_set = set(last_dates)
    daily_spot: dict[date, float] = {d: spot_sum[d] / max(spot_n[d], 1) for d in all_dates}

    candidates_by_date: dict[date, list[Candidate]] = {d: [] for d in last_dates}
    all_candidates_by_date: dict[date, list[Candidate]] = {d: [] for d in last_dates}

    with data_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("symbol", "")).upper() != str(args.symbol).upper():
                continue
            d = parse_date(row.get("date", ""))
            exp = parse_date(row.get("expiration", ""))
            if d is None or exp is None or d not in last_set:
                continue
            opt = norm_type(row.get("option_type", ""))
            if opt is None:
                continue
            if args.option_type != "both" and opt != args.option_type:
                continue
            strike = safe_float(row.get("strike_price"))
            premium = safe_float(row.get("market_price"))
            s0 = safe_float(row.get("S0"))
            vol = safe_float(row.get("volume")) or 0.0
            oi = safe_float(row.get("open_interest")) or 0.0
            sigma = norm_sigma(safe_float(row.get("sigma")))
            r = norm_rate(safe_float(row.get("r")))
            q = norm_rate(safe_float(row.get("q")))
            if None in (strike, premium, s0):
                continue
            if strike <= 0 or premium <= 0 or s0 <= 0:
                continue
            if vol < float(args.min_volume) or oi < float(args.min_open_interest):
                continue
            mny = s0 / strike
            if abs(mny - 1.0) > float(args.max_moneyness_dev):
                continue

            dte = (exp - d).days
            cand = Candidate(
                trade_date=d,
                expiration=exp,
                option_type=opt,
                strike=float(strike),
                premium=float(premium),
                s0=float(s0),
                volume=float(vol),
                open_interest=float(oi),
                contract_id=str(row.get("contractID", "")),
                dte=int(dte),
            )
            fair = model_fair_price(
                model_kind=model_kind,
                model=model,
                s0=float(s0),
                strike=float(strike),
                dte=int(dte),
                sigma=float(sigma),
                r=float(r),
                q=float(q),
                opt=opt,
            )
            cand.model_fair = fair
            cand.model_edge = (fair - float(premium)) if fair is not None else None
            if cand.model_edge is not None and cand.model_edge < float(args.min_model_edge):
                continue

            if dte > 0:
                all_candidates_by_date[d].append(cand)
            if abs(dte - int(args.target_dte)) > int(args.dte_tol):
                continue
            if abs(strike - s0) > float(args.max_strike_gap):
                continue
            candidates_by_date[d].append(cand)

    out_rows: list[dict[str, Any]] = []
    total_pnl = 0.0
    executed = 0
    skipped = 0

    for d in last_dates:
        spot_d = daily_spot.get(d)
        cands = candidates_by_date.get(d, [])
        used_relaxed = False
        if (not cands) and bool(args.force_daily_trade):
            relaxed_pool = all_candidates_by_date.get(d, [])
            if spot_d is not None and relaxed_pool:
                pick_relaxed = choose_one_relaxed(
                    relaxed_pool,
                    spot_d,
                    int(args.target_dte),
                    use_model_rank=bool(args.use_model_rank),
                )
                cands = [pick_relaxed]
                used_relaxed = True

        if spot_d is None or not cands:
            skipped += 1
            out_rows.append(
                {
                    "trade_date": d.isoformat(),
                    "status": "SKIP_NO_CANDIDATE",
                    "spot_trade": "" if spot_d is None else f"{spot_d:.6f}",
                    "contract_id": "",
                    "option_type": "",
                    "strike": "",
                    "premium": "",
                    "expiration": "",
                    "spot_expiry_used": "",
                    "intrinsic_expiry": "",
                    "pnl_per_contract": "",
                    "pnl_1lot": "",
                    "selection_mode": "none",
                    "model_fair": "",
                    "model_edge": "",
                }
            )
            continue

        pick = cands[0] if used_relaxed else choose_one(cands, spot_d, use_model_rank=bool(args.use_model_rank))

        settle_date = nearest_settle_date(all_dates, pick.expiration)
        used_exp_date = settle_date if settle_date is not None else pick.expiration
        s_exp = None if settle_date is None else daily_spot.get(settle_date)
        if s_exp is None:
            skipped += 1
            out_rows.append(
                {
                    "trade_date": d.isoformat(),
                    "status": "SKIP_NO_EXPIRY_SPOT",
                    "spot_trade": f"{spot_d:.6f}",
                    "contract_id": pick.contract_id,
                    "option_type": pick.option_type,
                    "strike": f"{pick.strike:.6f}",
                    "premium": f"{pick.premium:.6f}",
                    "expiration": pick.expiration.isoformat(),
                    "spot_expiry_used": "",
                    "intrinsic_expiry": "",
                    "pnl_per_contract": "",
                    "pnl_1lot": "",
                    "selection_mode": "relaxed" if used_relaxed else "strict",
                    "model_fair": "" if pick.model_fair is None else f"{pick.model_fair:.6f}",
                    "model_edge": "" if pick.model_edge is None else f"{pick.model_edge:.6f}",
                }
            )
            continue

        intrinsic = payoff(pick.option_type, s_exp, pick.strike)
        pnl_per_contract = intrinsic - pick.premium
        pnl_1lot = pnl_per_contract * float(args.contract_multiplier)
        total_pnl += pnl_1lot
        executed += 1

        out_rows.append(
            {
                "trade_date": d.isoformat(),
                "status": "EXECUTED",
                "spot_trade": f"{spot_d:.6f}",
                "contract_id": pick.contract_id,
                "option_type": pick.option_type,
                "strike": f"{pick.strike:.6f}",
                "premium": f"{pick.premium:.6f}",
                "expiration": pick.expiration.isoformat(),
                "spot_expiry_used": f"{s_exp:.6f} ({used_exp_date.isoformat()})",
                "intrinsic_expiry": f"{intrinsic:.6f}",
                "pnl_per_contract": f"{pnl_per_contract:.6f}",
                "pnl_1lot": f"{pnl_1lot:.6f}",
                "selection_mode": "relaxed" if used_relaxed else "strict",
                "model_fair": "" if pick.model_fair is None else f"{pick.model_fair:.6f}",
                "model_edge": "" if pick.model_edge is None else f"{pick.model_edge:.6f}",
            }
        )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()) if out_rows else ["trade_date", "status"])
        w.writeheader()
        w.writerows(out_rows)

    print("[OK] backtest complete.")
    print(f"[INFO] data_csv={data_csv}")
    print(f"[INFO] lookback_trade_days={len(last_dates)}")
    print(f"[INFO] executed_days={executed} skipped_days={skipped}")
    print(f"[INFO] total_pnl_1lot_each_day={total_pnl:.6f}")
    print(f"[INFO] avg_pnl_per_executed_day={(total_pnl / executed) if executed > 0 else float('nan'):.6f}")
    if model_kind is not None:
        print(f"[INFO] model_kind={model_kind} use_model_rank={bool(args.use_model_rank)}")
    print(f"[INFO] out_csv={out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
