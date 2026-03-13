#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _bs_d1_d2(*, s: float, k: float, t: float, r: float, q: float, sigma: float) -> tuple[float, float]:
    sqrt_t = math.sqrt(max(t, 1e-12))
    d1 = (math.log(max(s, 1e-12) / max(k, 1e-12)) + (r - q + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return d1, d2


def _bs_price(*, is_call: bool, s: float, k: float, t: float, r: float, q: float, sigma: float) -> float:
    if t <= 0.0:
        return max(0.0, (s - k) if is_call else (k - s))
    sigma = max(sigma, 1e-12)
    d1, d2 = _bs_d1_d2(s=s, k=k, t=t, r=r, q=q, sigma=sigma)
    df_r = math.exp(-r * t)
    df_q = math.exp(-q * t)
    if is_call:
        return s * df_q * _norm_cdf(d1) - k * df_r * _norm_cdf(d2)
    return k * df_r * _norm_cdf(-d2) - s * df_q * _norm_cdf(-d1)


def _bs_vega(*, s: float, k: float, t: float, r: float, q: float, sigma: float) -> float:
    if t <= 0.0:
        return 0.0
    d1, _ = _bs_d1_d2(s=s, k=k, t=t, r=r, q=q, sigma=max(sigma, 1e-12))
    return s * math.exp(-q * t) * _norm_pdf(d1) * math.sqrt(t)


def _bs_delta(*, is_call: bool, s: float, k: float, t: float, r: float, q: float, sigma: float) -> float:
    if t <= 0.0:
        if is_call:
            return 1.0 if s > k else 0.0
        return -1.0 if s < k else 0.0
    d1, _ = _bs_d1_d2(s=s, k=k, t=t, r=r, q=q, sigma=max(sigma, 1e-12))
    df_q = math.exp(-q * t)
    if is_call:
        return df_q * _norm_cdf(d1)
    return df_q * (_norm_cdf(d1) - 1.0)


def _iv_from_price(
    *,
    target_price: float,
    is_call: bool,
    s: float,
    k: float,
    t: float,
    r: float,
    q: float,
    low: float = 1e-4,
    high: float = 3.0,
    iters: int = 80,
) -> float | None:
    if target_price <= 0.0 or s <= 0.0 or k <= 0.0 or t <= 0.0:
        return None
    intrinsic = max(0.0, (s * math.exp(-q * t) - k * math.exp(-r * t)) if is_call else (k * math.exp(-r * t) - s * math.exp(-q * t)))
    if target_price < intrinsic - 1e-8:
        return None
    p_low = _bs_price(is_call=is_call, s=s, k=k, t=t, r=r, q=q, sigma=low)
    p_high = _bs_price(is_call=is_call, s=s, k=k, t=t, r=r, q=q, sigma=high)
    if target_price < p_low - 1e-8 or target_price > p_high + 1e-8:
        return None
    lo, hi = low, high
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        pm = _bs_price(is_call=is_call, s=s, k=k, t=t, r=r, q=q, sigma=mid)
        if pm > target_price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    return v if math.isfinite(v) else None


def _to_rate(v: float | None) -> float | None:
    if v is None:
        return None
    if abs(v) > 3.0:
        return v / 100.0
    return v


def _to_sigma(v: float | None) -> float | None:
    if v is None:
        return None
    if v > 3.0:
        return v / 100.0
    return v


def _parse_expiry(x: Any) -> date | None:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return datetime.strptime(s[:10], fmt).date()
        except Exception:
            pass
    try:
        return pd.to_datetime(s, errors="coerce").date()
    except Exception:
        return None


def _norm_option_type(x: Any) -> str | None:
    s = str(x or "").strip().lower()
    if not s:
        return None
    if s in {"1"}:
        return "call"
    if s in {"2"}:
        return "put"
    if "call" in s or s in {"c", "ca"}:
        return "call"
    if "put" in s or s in {"p", "po"}:
        return "put"
    return None


@dataclass
class MLPArtifact:
    model_name: str
    mean: list[float]
    std: list[float]
    w1: list[list[float]]
    b1: list[float]
    w2: list[float]
    b2: float


def load_mlp_artifact(model_path: Path) -> MLPArtifact:
    d = json.loads(model_path.read_text(encoding="utf-8"))
    model = d["model"]
    if str(model.get("kind", "")) != "mlp":
        raise ValueError(f"Unsupported model kind: {model.get('kind')}")
    return MLPArtifact(
        model_name=str(d.get("model_name", "mlp_direct")),
        mean=[float(x) for x in d["normalization"]["mean"]],
        std=[float(x) for x in d["normalization"]["std"]],
        w1=[[float(y) for y in row] for row in model["w1"]],
        b1=[float(x) for x in model["b1"]],
        w2=[float(x) for x in model["w2"]],
        b2=float(model["b2"]),
    )


def mlp_predict(x: list[float], m: MLPArtifact) -> float:
    z: list[float] = []
    for i, row in enumerate(m.w1):
        s = m.b1[i]
        for a, b in zip(row, x):
            s += a * b
        z.append(s if s > 0.0 else 0.0)
    y = m.b2
    for w, a in zip(m.w2, z):
        y += w * a
    return y


def _build_features(
    *,
    s0: float,
    strike: float,
    t_years: float,
    sigma: float,
    r: float,
    q: float,
    option_type: str,
    norm_mean: list[float],
    norm_std: list[float],
) -> list[float]:
    m = s0 / strike
    is_call = 1.0 if option_type == "call" else 0.0
    feat = [
        math.log(max(m, 1e-12)),
        t_years,
        sigma,
        r,
        q,
        is_call,
        math.sqrt(max(t_years, 0.0)),
        m,
    ]
    return [(feat[i] - norm_mean[i]) / norm_std[i] for i in range(len(feat))]


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def _to_canonical_options(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()

    col_code = _pick_col(out, ["option_code", "code_quote", "code", "contract_code"])
    col_type = _pick_col(out, ["option_type", "type", "option_side", "call_put"])
    col_strike = _pick_col(out, ["strike_price", "strike"])
    col_exp = _pick_col(out, ["expiry_date", "expiration", "maturity_date", "strike_time"])
    col_bid = _pick_col(out, ["bid_price_quote", "bid_price", "bid", "bid_price1"])
    col_ask = _pick_col(out, ["ask_price_quote", "ask_price", "ask", "ask_price1"])
    col_last = _pick_col(out, ["last_price_quote", "last_price", "last"])
    col_mark = _pick_col(out, ["mark_price_quote", "mark_price", "mark"])
    col_iv = _pick_col(out, ["implied_volatility_quote", "implied_volatility", "option_implied_volatility", "iv"])
    col_vol = _pick_col(out, ["volume_quote", "volume"])
    col_oi = _pick_col(out, ["open_interest_quote", "open_interest"])

    keep = pd.DataFrame(
        {
            "option_code": out[col_code].astype(str) if col_code else "",
            "option_type": out[col_type].map(_norm_option_type) if col_type else None,
            "strike_price": pd.to_numeric(out[col_strike], errors="coerce") if col_strike else math.nan,
            "expiration": out[col_exp].astype(str) if col_exp else "",
            "bid": pd.to_numeric(out[col_bid], errors="coerce") if col_bid else math.nan,
            "ask": pd.to_numeric(out[col_ask], errors="coerce") if col_ask else math.nan,
            "last": pd.to_numeric(out[col_last], errors="coerce") if col_last else math.nan,
            "mark": pd.to_numeric(out[col_mark], errors="coerce") if col_mark else math.nan,
            "iv": pd.to_numeric(out[col_iv], errors="coerce") if col_iv else math.nan,
            "volume": pd.to_numeric(out[col_vol], errors="coerce") if col_vol else 0.0,
            "open_interest": pd.to_numeric(out[col_oi], errors="coerce") if col_oi else 0.0,
        }
    )
    keep["option_type"] = keep["option_type"].fillna("")
    keep = keep[keep["option_type"].isin(["call", "put"])]
    keep = keep[keep["strike_price"].notna() & (keep["strike_price"] > 0)]
    keep["expiration_dt"] = keep["expiration"].map(_parse_expiry)
    keep = keep[keep["expiration_dt"].notna()]
    keep["option_code"] = keep["option_code"].fillna("").astype(str)
    keep = keep[keep["option_code"].str.len() >= 6]
    keep.reset_index(drop=True, inplace=True)
    return keep


def _futu_underlying(symbol: str) -> str:
    sym = symbol.strip().upper()
    if "." in sym:
        return sym
    return f"US.{sym}"


def fetch_from_futu(
    *,
    symbol: str,
    host: str,
    port: int,
    right: str,
    max_contracts: int,
    max_expiry_days: int,
) -> tuple[float | None, pd.DataFrame, str | None]:
    try:
        import futu  # type: ignore
    except Exception as e:
        return None, pd.DataFrame(), f"futu-api missing: {e}"

    code = _futu_underlying(symbol)
    start = date.today().isoformat()
    end = (date.today() + timedelta(days=int(max_expiry_days))).isoformat()
    quote_ctx = None
    try:
        quote_ctx = futu.OpenQuoteContext(host=host, port=int(port))
        spot = None
        try:
            ret, snap = quote_ctx.get_market_snapshot([code])
            if ret == futu.RET_OK and snap is not None and len(snap) > 0:
                spot = _safe_float(snap.iloc[0].get("last_price")) or _safe_float(snap.iloc[0].get("close_price"))
        except Exception:
            # Some accounts have option quote permission but no stock snapshot permission.
            spot = None

        option_type = None
        right_u = right.upper()
        if right_u in {"CALL", "PUT"}:
            option_type = getattr(futu.OptionType, right_u, None)
        else:
            option_type = getattr(futu.OptionType, "ALL", None)
        option_cond_type = getattr(futu.OptionCondType, "ALL", None)
        ret, chain = quote_ctx.get_option_chain(
            code,
            start=start,
            end=end,
            option_type=option_type,
            option_cond_type=option_cond_type,
        )
        if ret != futu.RET_OK or chain is None or len(chain) == 0:
            return spot, pd.DataFrame(), f"futu option_chain failed: {chain}"

        code_col = _pick_col(chain, ["code", "option_code", "contract_code"])
        if not code_col:
            return spot, pd.DataFrame(), "futu chain missing option code column"
        option_codes = [str(x) for x in chain[code_col].head(int(max_contracts)).tolist()]

        q_ret, q_df = None, None
        try:
            q_ret, q_df = quote_ctx.get_option_quote(option_codes)
        except Exception:
            q_ret, q_df = quote_ctx.get_market_snapshot(option_codes)
        if q_ret != futu.RET_OK or q_df is None:
            merged = chain.copy()
        else:
            q_code_col = _pick_col(q_df, ["code", "option_code", "contract_code"])
            if q_code_col:
                merged = chain.merge(
                    q_df,
                    left_on=code_col,
                    right_on=q_code_col,
                    how="left",
                    suffixes=("", "_quote"),
                )
            else:
                merged = chain.copy()
        return spot, _to_canonical_options(merged), None
    except Exception as e:
        return None, pd.DataFrame(), f"futu fetch exception: {e}"
    finally:
        if quote_ctx is not None:
            try:
                quote_ctx.close()
            except Exception:
                pass


def _load_dotenv() -> None:
    env = Path(__file__).resolve().parents[2] / ".env"
    if not env.exists():
        return
    for raw in env.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip().strip("'").strip('"')
        if k and k not in os.environ:
            os.environ[k] = v


def _fetch_alpha_quote(symbol: str, api_key: str) -> float | None:
    params = {"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": api_key}
    resp = requests.get("https://www.alphavantage.co/query", params=params, timeout=20)
    resp.raise_for_status()
    payload = resp.json()
    q = payload.get("Global Quote", {})
    return _safe_float(q.get("05. price"))


def _fetch_alpha_options(symbol: str, api_key: str, date_str: str) -> pd.DataFrame:
    params = {"function": "HISTORICAL_OPTIONS", "symbol": symbol, "date": date_str, "apikey": api_key}
    resp = requests.get("https://www.alphavantage.co/query", params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    data = payload.get("data")
    if not isinstance(data, list):
        return pd.DataFrame()
    return _to_canonical_options(pd.DataFrame(data))


def fetch_from_alpha_vantage(
    *,
    symbol: str,
    api_key: str,
    right: str,
    max_contracts: int,
) -> tuple[float | None, pd.DataFrame, str | None]:
    if not api_key:
        return None, pd.DataFrame(), "AlphaVantage API key missing"
    try:
        spot = _fetch_alpha_quote(symbol, api_key)
        options = pd.DataFrame()
        for lag in range(0, 7):
            d = (date.today() - timedelta(days=lag)).isoformat()
            options = _fetch_alpha_options(symbol, api_key, d)
            if not options.empty:
                break
        if options.empty:
            return spot, pd.DataFrame(), "AlphaVantage HISTORICAL_OPTIONS empty in last 7 days"
        if right.upper() in {"CALL", "PUT"}:
            options = options[options["option_type"] == right.lower()]
        options = options.head(int(max_contracts)).copy()
        return spot, options, None
    except Exception as e:
        return None, pd.DataFrame(), f"alpha fetch exception: {e}"


def compute_opportunities(
    *,
    options_df: pd.DataFrame,
    spot: float,
    model: MLPArtifact,
    today: date,
    default_r: float,
    default_q: float,
    default_sigma: float,
    fee_per_contract: float,
    slippage_bps: float,
    spread_cross_ratio: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in options_df.iterrows():
        strike = _safe_float(row.get("strike_price"))
        exp = row.get("expiration_dt")
        opt_type = _norm_option_type(row.get("option_type"))
        if strike is None or strike <= 0 or exp is None or opt_type is None:
            continue
        t_days = (exp - today).days
        if t_days <= 0:
            continue
        t = t_days / 365.0

        bid = _safe_float(row.get("bid"))
        ask = _safe_float(row.get("ask"))
        mark = _safe_float(row.get("mark"))
        last = _safe_float(row.get("last"))
        iv = _to_sigma(_safe_float(row.get("iv"))) or float(default_sigma)
        r = float(default_r)
        q = float(default_q)
        if bid is not None and ask is not None and ask >= bid > 0:
            mid = 0.5 * (bid + ask)
            spread = ask - bid
        else:
            mid = mark if (mark is not None and mark > 0) else last
            if mid is None or mid <= 0:
                continue
            spread = max(mid * 0.02, 0.01)
        x = _build_features(
            s0=spot,
            strike=strike,
            t_years=t,
            sigma=iv,
            r=r,
            q=q,
            option_type=opt_type,
            norm_mean=model.mean,
            norm_std=model.std,
        )
        fair = mlp_predict(x, model)
        # Basic no-arbitrage sanity filter: remove obviously mis-mapped quotes.
        if opt_type == "call" and mid > spot * 1.02:
            continue
        if opt_type == "put" and mid > strike * 1.05:
            continue
        mispricing = mid - fair
        is_call = opt_type == "call"
        iv_mkt = _iv_from_price(
            target_price=mid,
            is_call=is_call,
            s=spot,
            k=strike,
            t=t,
            r=r,
            q=q,
        )
        iv_fair = _iv_from_price(
            target_price=max(fair, 1e-8),
            is_call=is_call,
            s=spot,
            k=strike,
            t=t,
            r=r,
            q=q,
        )
        vol_edge = None
        vol_action = "N/A"
        if iv_mkt is not None and iv_fair is not None:
            vol_edge = iv_mkt - iv_fair
            if vol_edge > 0:
                vol_action = "SELL_VOL"
            elif vol_edge < 0:
                vol_action = "BUY_VOL"
            else:
                vol_action = "FAIR_VOL"
        use_iv = iv_mkt if iv_mkt is not None else iv
        vega = _bs_vega(s=spot, k=strike, t=t, r=r, q=q, sigma=max(use_iv, 1e-4))
        delta = _bs_delta(is_call=is_call, s=spot, k=strike, t=t, r=r, q=q, sigma=max(use_iv, 1e-4))
        cost = float(fee_per_contract) + (float(slippage_bps) / 10000.0) * mid + float(spread_cross_ratio) * spread
        edge = abs(mispricing) - cost
        action = "BUY_UNDERVALUE" if mispricing < 0 else "SELL_OVERVALUE"
        volume = _safe_float(row.get("volume")) or 0.0
        oi = _safe_float(row.get("open_interest")) or 0.0
        rows.append(
            {
                "option_code": str(row.get("option_code", "")),
                "option_type": opt_type,
                "expiration": exp.isoformat(),
                "ttm_days": t_days,
                "strike_price": strike,
                "spot": spot,
                "mid_price": mid,
                "model_fair": fair,
                "mispricing": mispricing,
                "abs_mispricing": abs(mispricing),
                "iv_mkt": iv_mkt,
                "iv_fair": iv_fair,
                "vol_edge": vol_edge,
                "vol_action": vol_action,
                "est_cost": cost,
                "edge": edge,
                "action": action,
                "bid": bid,
                "ask": ask,
                "spread": spread,
                "iv": iv,
                "vega": vega,
                "delta": delta,
                "r": r,
                "q": q,
                "volume": volume,
                "open_interest": oi,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["edge", "abs_mispricing"], ascending=[False, False]).reset_index(drop=True)
    return out


def compute_parity_deviation(df: pd.DataFrame, r: float, q: float) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    g = df.copy()
    g["pair_key"] = g["expiration"].astype(str) + "|" + g["strike_price"].round(4).astype(str)
    calls = g[g["option_type"] == "call"][["pair_key", "expiration", "ttm_days", "strike_price", "mid_price", "spot"]]
    puts = g[g["option_type"] == "put"][["pair_key", "mid_price"]]
    c = calls.rename(columns={"mid_price": "call_mid"})
    p = puts.rename(columns={"mid_price": "put_mid"})
    m = c.merge(p, on="pair_key", how="inner")
    if m.empty:
        return pd.DataFrame()
    t = m["ttm_days"] / 365.0
    rhs = m["spot"] * (pd.Series([-q] * len(m)) * t).map(math.exp) - m["strike_price"] * (pd.Series([-r] * len(m)) * t).map(math.exp)
    m["parity_residual"] = (m["call_mid"] - m["put_mid"]) - rhs
    m["parity_abs"] = m["parity_residual"].abs()
    return m.sort_values("parity_abs", ascending=False).reset_index(drop=True)


def run_app() -> None:
    _load_dotenv()
    st.set_page_config(page_title="JPM Option Arbitrage Panel", layout="wide")
    st.title("JPM Option Arbitrage Panel")

    with st.sidebar:
        st.header("Data / Model")
        symbol = st.text_input("Underlying symbol", value="JPM").strip().upper()
        model_path = st.text_input("MLP model path", value=str((Path.cwd() / "models" / "mlp_direct.json").resolve()))
        source = st.selectbox(
            "Data source",
            ["Auto", "Hybrid (Futu options + Alpha spot)", "Futu", "AlphaVantage"],
            index=1,
        )
        right = st.selectbox("Option right", ["ALL", "CALL", "PUT"], index=0)
        max_contracts = st.slider("Max contracts", min_value=20, max_value=800, value=200, step=20)
        spot_override = st.number_input("Manual spot override (0=off)", value=0.0, step=0.1, format="%.4f")
        mode = st.selectbox("Arbitrage mode", ["Price edge", "Vol edge", "Both"], index=2)

        st.header("Rates / Vol")
        default_r = st.number_input("Risk-free rate r (cont.)", value=0.045, step=0.005, format="%.4f")
        default_q = st.number_input("Dividend yield q (cont.)", value=0.030, step=0.005, format="%.4f")
        default_sigma = st.number_input("Fallback sigma", value=0.25, step=0.01, format="%.4f")

        st.header("Execution Cost")
        fee = st.number_input("Fee per contract", value=0.02, step=0.01, format="%.4f")
        slippage_bps = st.number_input("Slippage (bps of mid)", value=5.0, step=1.0, format="%.1f")
        spread_cross_ratio = st.slider("Spread crossing ratio", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        min_edge = st.number_input("Min edge filter", value=0.05, step=0.01, format="%.4f")
        min_oi = st.number_input("Min open interest", value=0.0, step=10.0, format="%.0f")
        min_vol = st.number_input("Min volume", value=0.0, step=10.0, format="%.0f")
        min_abs_vol_edge = st.number_input("Min |vol_edge|", value=0.01, step=0.005, format="%.4f")

        st.header("Futu")
        futu_host = st.text_input("FUTU_HOST", value=os.getenv("FUTU_HOST", "127.0.0.1"))
        futu_port = st.number_input("FUTU_PORT", value=int(os.getenv("FUTU_PORT", "11111")), step=1)
        max_expiry_days = st.slider("Futu max expiry days", min_value=7, max_value=365, value=120, step=7)

        st.header("AlphaVantage")
        av_key = st.text_input("ALPHAVANTAGE_API_KEY", value=os.getenv("ALPHAVANTAGE_API_KEY", ""), type="password")

        refresh = st.button("Refresh")

    if not refresh:
        st.info("Set parameters in sidebar and click `Refresh`.")
        return

    model_file = Path(model_path)
    if not model_file.exists():
        st.error(f"Model file not found: {model_file}")
        return
    try:
        model = load_mlp_artifact(model_file)
    except Exception as e:
        st.error(f"Load model failed: {e}")
        return

    fetch_errors: list[str] = []
    spot: float | None = None
    options_df = pd.DataFrame()
    used_source = ""

    if source == "Hybrid (Futu options + Alpha spot)":
        used_source = "Hybrid"
        futu_spot, futu_opt, err_f = fetch_from_futu(
            symbol=symbol,
            host=str(futu_host),
            port=int(futu_port),
            right=right,
            max_contracts=int(max_contracts),
            max_expiry_days=int(max_expiry_days),
        )
        if err_f:
            fetch_errors.append(f"Futu: {err_f}")
        options_df = futu_opt
        # Prefer Alpha spot (even delayed), fallback to Futu spot if available.
        av_spot = None
        if str(av_key).strip():
            try:
                av_spot = _fetch_alpha_quote(symbol, str(av_key).strip())
            except Exception as e:
                fetch_errors.append(f"AlphaVantage spot: {e}")
        spot = av_spot if av_spot is not None else futu_spot
    elif source in {"Auto", "Futu"}:
        used_source = "Futu"
        spot, options_df, err = fetch_from_futu(
            symbol=symbol,
            host=str(futu_host),
            port=int(futu_port),
            right=right,
            max_contracts=int(max_contracts),
            max_expiry_days=int(max_expiry_days),
        )
        if err:
            fetch_errors.append(f"Futu: {err}")

    if (source in {"Auto", "AlphaVantage"}) and (options_df.empty or spot is None):
        used_source = "AlphaVantage"
        spot, options_df, err = fetch_from_alpha_vantage(
            symbol=symbol,
            api_key=str(av_key),
            right=right,
            max_contracts=int(max_contracts),
        )
        if err:
            fetch_errors.append(f"AlphaVantage: {err}")

    if float(spot_override) > 0:
        spot = float(spot_override)

    if options_df.empty or spot is None:
        st.error("No usable market data fetched.")
        if fetch_errors:
            st.code("\n".join(fetch_errors))
        return

    opp = compute_opportunities(
        options_df=options_df,
        spot=float(spot),
        model=model,
        today=date.today(),
        default_r=float(default_r),
        default_q=float(default_q),
        default_sigma=float(default_sigma),
        fee_per_contract=float(fee),
        slippage_bps=float(slippage_bps),
        spread_cross_ratio=float(spread_cross_ratio),
    )
    if opp.empty:
        st.warning("No valid option rows after normalization.")
        return

    filt = opp[(opp["edge"] >= float(min_edge)) & (opp["open_interest"] >= float(min_oi)) & (opp["volume"] >= float(min_vol))].copy()
    filt_vol = opp[
        (opp["vol_edge"].notna())
        & (opp["open_interest"] >= float(min_oi))
        & (opp["volume"] >= float(min_vol))
        & (opp["vol_edge"].abs() >= float(min_abs_vol_edge))
    ].copy()
    if mode == "Price edge":
        show_df = filt.copy()
    elif mode == "Vol edge":
        show_df = filt_vol.copy()
    else:
        show_df = filt[(filt["vol_edge"].abs() >= float(min_abs_vol_edge)) | (filt["vol_edge"].isna())].copy()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Data Source", used_source)
    c2.metric("Spot", f"{spot:.4f}")
    c3.metric("Rows Fetched", f"{len(opp)}")
    c4.metric("Arb Opportunities", f"{len(show_df)}")

    c5, c6 = st.columns(2)
    c5.metric("Price-Edge Sum", f"{filt['edge'].sum():.4f}" if not filt.empty else "0.0000")
    c6.metric("Mean |Vol Edge|", f"{filt_vol['vol_edge'].abs().mean():.4f}" if not filt_vol.empty else "0.0000")

    st.subheader("Top Opportunities")
    show_cols = [
        "option_code",
        "option_type",
        "expiration",
        "ttm_days",
        "strike_price",
        "mid_price",
        "model_fair",
        "mispricing",
        "est_cost",
        "edge",
        "action",
        "iv_mkt",
        "iv_fair",
        "vol_edge",
        "vol_action",
        "vega",
        "delta",
        "bid",
        "ask",
        "spread",
        "iv",
        "volume",
        "open_interest",
    ]
    sort_col = "edge" if mode == "Price edge" else "vol_edge"
    ascending = False
    if mode == "Vol edge":
        show_df = show_df.reindex(show_df["vol_edge"].abs().sort_values(ascending=False).index)
    else:
        show_df = show_df.sort_values(sort_col, ascending=ascending)
    st.dataframe(show_df[show_cols].head(200), use_container_width=True, hide_index=True)

    csv_bytes = show_df[show_cols].to_csv(index=False).encode("utf-8")
    st.download_button("Download Opportunities CSV", data=csv_bytes, file_name=f"{symbol}_arb_opportunities.csv", mime="text/csv")

    st.subheader("Mispricing vs Edge")
    st.scatter_chart(show_df[["abs_mispricing", "edge"]].rename(columns={"abs_mispricing": "abs_mispricing", "edge": "edge"}))
    if not show_df["vol_edge"].dropna().empty:
        st.subheader("Vol Edge Distribution")
        st.bar_chart(show_df["vol_edge"].dropna().head(120))

    st.subheader("Put-Call Parity Monitor")
    parity = compute_parity_deviation(opp, r=float(default_r), q=float(default_q))
    if parity.empty:
        st.info("No call-put pairs found at same strike/expiry.")
    else:
        st.dataframe(parity[["expiration", "ttm_days", "strike_price", "call_mid", "put_mid", "parity_residual", "parity_abs"]].head(80), use_container_width=True, hide_index=True)

    if fetch_errors:
        st.caption("Fetch warnings:\n" + "\n".join(fetch_errors))


if __name__ == "__main__":
    run_app()
