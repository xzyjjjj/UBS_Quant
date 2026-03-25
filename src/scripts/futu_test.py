#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
import inspect


def _parse_args() -> argparse.Namespace:
    today = date.today()
    default_start = (today - timedelta(days=30)).isoformat()
    default_end = today.isoformat()

    p = argparse.ArgumentParser(
        description="Smoke test for Futu OpenAPI (OpenD) data access using HK stocks"
    )
    p.add_argument("--host", default=os.getenv("FUTU_HOST", "127.0.0.1"), help="OpenD host")
    p.add_argument("--port", type=int, default=int(os.getenv("FUTU_PORT", "11111")), help="OpenD port")
    p.add_argument(
        "--codes",
        default="HK.00700",
        help="Comma-separated codes (default: HK.00700)",
    )
    p.add_argument("--start", default=default_start, help="Kline start date (YYYY-MM-DD)")
    p.add_argument("--end", default=default_end, help="Kline end date (YYYY-MM-DD)")
    p.add_argument(
        "--ktype",
        default="K_DAY",
        choices=("K_1M", "K_5M", "K_15M", "K_30M", "K_60M", "K_DAY", "K_WEEK", "K_MON"),
        help="Kline type (default: K_DAY)",
    )
    p.add_argument("--max-count", type=int, default=500, help="Max bars per request page (default: 500)")
    p.add_argument(
        "--out-dir",
        default="output/futu",
        help="Directory to save CSV outputs (default: output/futu)",
    )
    p.add_argument(
        "--calc-sigma",
        action="store_true",
        help="Compute annualized historical volatility from kline (default: off)",
    )
    p.add_argument(
        "--sigma-window",
        type=int,
        default=252,
        help="Number of returns used for hist vol (default: 252)",
    )
    p.add_argument(
        "--trading-days",
        type=int,
        default=252,
        help="Trading days per year for annualization (default: 252)",
    )
    p.add_argument(
        "--calc-iv",
        action="store_true",
        help="Try computing implied vol from an option quote (default: off)",
    )
    p.add_argument(
        "--price-vanilla",
        action="store_true",
        help="Batch price vanilla options from option_chain (BS theoretical + market quotes when available)",
    )
    p.add_argument(
        "--vanilla-expiry",
        default="",
        help="Only price options for this expiry (YYYY-MM-DD). Default: all expiries returned by chain.",
    )
    p.add_argument(
        "--vanilla-right",
        default="ALL",
        choices=("ALL", "CALL", "PUT"),
        help="Only price CALL/PUT when available (default: ALL)",
    )
    p.add_argument(
        "--vanilla-max",
        type=int,
        default=60,
        help="Max contracts to quote/price (default: 60; protects against huge chains)",
    )
    p.add_argument(
        "--dump-option-expiries",
        action="store_true",
        help="Fetch and dump option expiration dates for underlying (default: off)",
    )
    p.add_argument(
        "--dump-option-chain",
        action="store_true",
        help="Fetch and dump option chain for underlying (default: off)",
    )
    p.add_argument(
        "--chain-start",
        default="",
        help="Option chain start expiry date (YYYY-MM-DD). Default: auto/none",
    )
    p.add_argument(
        "--chain-end",
        default="",
        help="Option chain end expiry date (YYYY-MM-DD). Default: auto/none",
    )
    p.add_argument(
        "--chain-right",
        default="ALL",
        choices=("ALL", "CALL", "PUT"),
        help="Filter option chain by right when supported (default: ALL)",
    )
    p.add_argument(
        "--chain-cond",
        default="ALL",
        help="Option condition filter when supported (e.g. ALL/ITM/OTM; default: ALL)",
    )
    p.add_argument(
        "--option-code",
        default="",
        help="Option code (e.g., HK.XXXXXX). If set, skip option chain discovery.",
    )
    p.add_argument(
        "--strike",
        type=float,
        default=float(os.getenv("FUTU_STRIKE", "0") or 0),
        help="Strike used for IV solving (same currency as underlying/option)",
    )
    p.add_argument(
        "--expiry",
        default=os.getenv("FUTU_EXPIRY", ""),
        help="Option expiry date (YYYY-MM-DD) used for IV solving/discovery",
    )
    p.add_argument(
        "--right",
        default="CALL",
        choices=("CALL", "PUT"),
        help="Option right when solving IV (default: CALL)",
    )
    p.add_argument(
        "--r",
        type=float,
        default=float(os.getenv("FUTU_R", "0") or 0),
        help="Risk-free rate for IV solving (annualized, continuously compounded; default: 0)",
    )
    p.add_argument(
        "--q",
        type=float,
        default=float(os.getenv("FUTU_Q", "0") or 0),
        help="Dividend yield for IV solving (annualized, continuously compounded; default: 0)",
    )
    p.add_argument(
        "--use-mid",
        action="store_true",
        help="Use mid price (bid/ask) for IV when available (default: off)",
    )
    p.add_argument(
        "--sigma",
        type=float,
        default=float(os.getenv("FUTU_SIGMA", "0") or 0),
        help="Vol used for BSM theoretical pricing (annualized). Default: use hist_sigma if computed.",
    )
    p.add_argument(
        "--subscribe-rt",
        action="store_true",
        help="Also try subscribing to real-time quote push for a few seconds",
    )
    p.add_argument(
        "--rt-seconds",
        type=int,
        default=10,
        help="Seconds to wait for push messages when --subscribe-rt (default: 10)",
    )
    return p.parse_args()


def _require_futu() -> "object":
    try:
        import futu  # type: ignore

        return futu
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: futu-api. Install it with `pip install futu-api` "
            "(and make sure Futu OpenD is running)."
        ) from e


@dataclass(frozen=True)
class FetchResult:
    ok: bool
    data: "object"
    error: str | None = None


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        if math.isfinite(float(x)):
            return float(x)
        return None
    s = str(x).strip()
    if not s or s.upper() == "N/A":
        return None
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _print_preview(title: str, obj: "object", *, max_rows: int = 5) -> None:
    print(f"\n== {title} ==")
    try:
        shape = getattr(obj, "shape", None)
        cols = getattr(obj, "columns", None)
        print(f"shape={shape}")
        if cols is not None:
            cols_list = list(cols)
            print(f"columns={cols_list[:12]}{' ...' if len(cols_list) > 12 else ''}")
        if hasattr(obj, "head"):
            print(obj.head(max_rows))
        else:
            print(obj)
    except Exception:
        print(obj)


def _save_csv(df: "object", path: Path) -> None:
    try:
        import pandas as pd  # type: ignore

        if isinstance(df, pd.DataFrame):
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)
            print(f"saved: {path}")
    except Exception:
        # Best-effort only; printing is already done.
        return


def _extract_scalar(df: Any, col: str) -> Any:
    try:
        if df is None:
            return None
        if not hasattr(df, "__getitem__"):
            return None
        s = df[col]
        if hasattr(s, "iloc"):
            return s.iloc[0]
        if isinstance(s, list) and s:
            return s[0]
        return s
    except Exception:
        return None


def _compute_hist_sigma(kline_df: Any, *, window: int, trading_days: int) -> dict[str, Any]:
    try:
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore
    except Exception as e:
        return {"ok": False, "error": f"missing numpy/pandas: {e}"}

    if not isinstance(kline_df, pd.DataFrame) or kline_df.empty:
        return {"ok": False, "error": "kline dataframe is empty"}

    df = kline_df.copy()
    if "time_key" in df.columns:
        try:
            df = df.sort_values("time_key")
        except Exception:
            pass

    if "close" not in df.columns:
        return {"ok": False, "error": "missing close column in kline"}

    px = pd.to_numeric(df["close"], errors="coerce").dropna()
    if len(px) < 3:
        return {"ok": False, "error": "not enough close prices for returns"}

    rets = np.log(px).diff().dropna()
    if window and window > 0:
        rets = rets.tail(int(window))
    if len(rets) < 2:
        return {"ok": False, "error": "not enough returns after windowing"}

    sigma_daily = float(rets.std(ddof=1))
    sigma_ann = sigma_daily * math.sqrt(float(trading_days))

    return {
        "ok": True,
        "n_prices": int(len(px)),
        "n_returns": int(len(rets)),
        "sigma_daily": sigma_daily,
        "sigma_annualized": sigma_ann,
        "trading_days": int(trading_days),
        "window": int(window),
    }


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_vanilla_price(
    *,
    is_call: bool,
    s: float,
    k: float,
    t: float,
    r: float,
    q: float,
    sigma: float,
) -> float:
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
    if t <= 0:
        return None
    if price <= 0:
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


def _try_call(ctx: Any, method_names: tuple[str, ...], *args: Any, **kwargs: Any) -> tuple[Any, Any] | None:
    last: Exception | None = None
    for name in method_names:
        if not hasattr(ctx, name):
            continue
        fn = getattr(ctx, name)
        # Filter kwargs by signature to support multiple futu-api versions.
        call_kwargs = kwargs
        try:
            sig = inspect.signature(fn)
            allowed = set(sig.parameters.keys())
            call_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
        except Exception:
            call_kwargs = kwargs
        try:
            return fn(*args, **call_kwargs)
        except TypeError as e:
            last = e
            continue
        except Exception:
            raise
    if last is not None:
        raise last
    return None


def _fetch_option_quote(quote_ctx: Any, futu: Any, option_code: str) -> tuple[bool, Any, str | None]:
    # Try dedicated option quote API first; fall back to market snapshot.
    try:
        out = _try_call(quote_ctx, ("get_option_quote", "request_option_quote"), [option_code])
        if out is not None:
            ret, df = out
            if ret == futu.RET_OK:
                return True, df, None
            return False, df, str(df)
    except Exception:
        pass

    ret, df = quote_ctx.get_market_snapshot([option_code])
    if ret == futu.RET_OK:
        return True, df, None
    return False, df, str(df)


def _parse_yyyy_mm_dd(s: str) -> date | None:
    s2 = (s or "").strip()
    if not s2:
        return None
    try:
        return datetime.strptime(s2, "%Y-%m-%d").date()
    except Exception:
        return None


def _enum_or_none(obj: Any, enum_name: str, member: str) -> Any:
    try:
        enum = getattr(obj, enum_name)
        return getattr(enum, member)
    except Exception:
        return None


def _fetch_option_expiries(quote_ctx: Any, futu: Any, underlying: str) -> tuple[bool, Any, str | None]:
    out = _try_call(quote_ctx, ("get_option_expiration_date",), underlying)
    if out is None:
        return False, None, "method get_option_expiration_date not found"
    ret, df = out
    if ret == futu.RET_OK:
        return True, df, None
    return False, df, str(df)


def _fetch_option_chain(
    quote_ctx: Any,
    futu: Any,
    underlying: str,
    *,
    start: str | None,
    end: str | None,
    chain_right: str,
    chain_cond: str,
) -> tuple[bool, Any, str | None]:
    option_type = None
    if str(chain_right).upper() in ("CALL", "PUT"):
        option_type = _enum_or_none(futu, "OptionType", str(chain_right).upper())
    if str(chain_right).upper() == "ALL":
        option_type = _enum_or_none(futu, "OptionType", "ALL")

    option_cond_type = _enum_or_none(futu, "OptionCondType", str(chain_cond).upper()) or _enum_or_none(
        futu, "OptionCondType", "ALL"
    )

    kwargs: dict[str, Any] = {
        "start": start,
        "end": end,
        "option_type": option_type,
        "option_cond_type": option_cond_type,
    }

    # Some versions expose index_option_type; keep it optional.
    idx_type = _enum_or_none(futu, "IndexOptionType", "NORMAL")
    if idx_type is not None:
        kwargs["index_option_type"] = idx_type

    out = _try_call(quote_ctx, ("get_option_chain",), underlying, **kwargs)
    if out is None:
        return False, None, "method get_option_chain not found"
    ret, df = out
    if ret == futu.RET_OK:
        return True, df, None
    return False, df, str(df)


def _pick_option_from_chain(chain_df: Any, *, s0: float | None, strike: float, right: str) -> dict[str, Any]:
    """
    Best-effort pick an option contract from chain:
    - prefer strike closest to `strike` if strike>0 else closest to s0 (ATM)
    - prefer matching `right` when possible
    Returns dict with ok, option_code, strike_price, expiry_date, row (optional).
    """
    try:
        import pandas as pd  # type: ignore

        if not isinstance(chain_df, pd.DataFrame) or chain_df.empty:
            return {"ok": False, "error": "empty chain dataframe"}

        df = chain_df.copy()

        # Identify columns
        code_col = None
        for c in ("option_code", "code", "contract_code", "security_code"):
            if c in df.columns:
                code_col = c
                break
        if code_col is None:
            return {"ok": False, "error": "cannot find option code column in chain"}

        strike_col = "strike_price" if "strike_price" in df.columns else None
        expiry_col = None
        for c in ("expiry_date", "maturity_date", "strike_time", "time"):
            if c in df.columns:
                expiry_col = c
                break

        # Right filter if possible
        if str(right).upper() in ("CALL", "PUT"):
            for c in ("option_type", "type", "call_put", "option_side"):
                if c in df.columns:
                    df2 = df[df[c].astype(str).str.upper().str.contains(str(right).upper())]
                    if not df2.empty:
                        df = df2
                    break

        # Strike selection
        target = strike if strike and strike > 0 else (s0 if s0 is not None else None)
        if target is not None and strike_col is not None:
            sp = pd.to_numeric(df[strike_col], errors="coerce")
            df = df.assign(_strike_num=sp)
            df = df.dropna(subset=["_strike_num"])
            if not df.empty:
                df = df.assign(_dist=(df["_strike_num"] - float(target)).abs()).sort_values("_dist")

        row = df.iloc[0]
        option_code = str(row[code_col])
        strike_price = float(row[strike_col]) if strike_col and _safe_float(row[strike_col]) is not None else None
        expiry_date = str(row[expiry_col]) if expiry_col else None
        return {
            "ok": True,
            "option_code": option_code,
            "strike_price": strike_price,
            "expiry_date": expiry_date,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _to_date(v: Any) -> date | None:
    if v is None:
        return None
    if isinstance(v, date) and not isinstance(v, datetime):
        return v
    s = str(v).strip()
    if not s:
        return None
    return _parse_yyyy_mm_dd(s)


def _build_option_universe(chain_df: Any, *, expiry: str, right: str, limit: int) -> tuple[list[str], Any]:
    try:
        import pandas as pd  # type: ignore

        if not isinstance(chain_df, pd.DataFrame) or chain_df.empty:
            return [], chain_df

        df = chain_df.copy()
        if expiry:
            for col in ("strike_time", "expiry_date", "maturity_date"):
                if col in df.columns:
                    df = df[df[col].astype(str) == str(expiry)]
                    break

        if str(right).upper() in ("CALL", "PUT"):
            for col in ("option_type", "type", "call_put", "option_side"):
                if col in df.columns:
                    df = df[df[col].astype(str).str.upper().str.contains(str(right).upper())]
                    break

        code_col = None
        for col in ("code", "option_code", "contract_code", "security_code"):
            if col in df.columns:
                code_col = col
                break
        if code_col is None:
            return [], df

        codes = [str(x) for x in df[code_col].head(int(limit)).tolist()]
        return codes, df
    except Exception:
        return [], chain_df


def _fetch_option_quotes(quote_ctx: Any, futu: Any, option_codes: list[str]) -> tuple[bool, Any, str | None]:
    if not option_codes:
        return False, None, "empty option_codes"
    try:
        out = _try_call(quote_ctx, ("get_option_quote", "request_option_quote"), option_codes)
        if out is not None:
            ret, df = out
            if ret == futu.RET_OK:
                return True, df, None
            return False, df, str(df)
    except Exception:
        pass

    # Fallback: snapshot for option codes (may not include all option-specific columns).
    ret, df = quote_ctx.get_market_snapshot(option_codes)
    if ret == futu.RET_OK:
        return True, df, None
    return False, df, str(df)


def _merge_chain_and_quotes(chain_df: Any, quote_df: Any) -> Any:
    try:
        import pandas as pd  # type: ignore

        if not isinstance(chain_df, pd.DataFrame) or chain_df.empty:
            return chain_df
        if not isinstance(quote_df, pd.DataFrame) or quote_df.empty:
            return chain_df

        chain = chain_df.copy()
        quotes = quote_df.copy()

        chain_code_col = "code" if "code" in chain.columns else ("option_code" if "option_code" in chain.columns else None)
        quote_code_col = "code" if "code" in quotes.columns else ("option_code" if "option_code" in quotes.columns else None)
        if chain_code_col is None or quote_code_col is None:
            return chain_df

        dup_cols = [c for c in quotes.columns if c in chain.columns and c != quote_code_col]
        if dup_cols:
            quotes = quotes.drop(columns=dup_cols)

        return chain.merge(quotes, how="left", left_on=chain_code_col, right_on=quote_code_col)
    except Exception:
        return chain_df


def main() -> int:
    args = _parse_args()
    quote_ctx = None
    try:
        futu = _require_futu()

        codes = [c.strip() for c in str(args.codes).split(",") if c.strip()]
        if not codes:
            raise SystemExit("No codes provided.")

        out_dir = Path(args.out_dir)

        quote_ctx = futu.OpenQuoteContext(host=str(args.host), port=int(args.port))

        # 1) Snapshot (fast sanity check)
        ret, snap = quote_ctx.get_market_snapshot(codes)
        if ret == futu.RET_OK:
            _print_preview("market_snapshot", snap)
            _save_csv(snap, out_dir / "snapshot.csv")
        else:
            print(f"snapshot error: {snap}")

        # 2) Historical kline (common for model/backtest)
        ktype = getattr(futu.KLType, str(args.ktype))
        code0 = codes[0]
        all_pages = []
        page_req_key = None
        while True:
            ret, kl, page_req_key = quote_ctx.request_history_kline(
                code0,
                start=str(args.start),
                end=str(args.end),
                ktype=ktype,
                max_count=int(args.max_count),
                page_req_key=page_req_key,
                session=futu.Session.ALL,
            )
            if ret != futu.RET_OK:
                print(f"kline error: {kl}")
                break
            all_pages.append(kl)
            if page_req_key is None:
                break

        if all_pages:
            try:
                import pandas as pd  # type: ignore

                kline_df = pd.concat(all_pages, ignore_index=True)
            except Exception:
                # Fallback: keep the last page if concat isn't available.
                kline_df = all_pages[-1]

            _print_preview(f"history_kline ({code0})", kline_df)
            _save_csv(kline_df, out_dir / f"kline_{code0.replace('.', '_')}.csv")
        else:
            kline_df = None

        metrics: dict[str, Any] = {
            "asof": datetime.now().isoformat(timespec="seconds"),
            "codes": codes,
            "kline_code": code0,
            "start": str(args.start),
            "end": str(args.end),
            "ktype": str(args.ktype),
        }

        # 2a) Option expiries / chain dump (vanilla options)
        if bool(args.dump_option_expiries) or bool(args.dump_option_chain):
            ok_e, exp_df, exp_err = _fetch_option_expiries(quote_ctx, futu, code0)
            if ok_e:
                _print_preview(f"option_expiration_date ({code0})", exp_df, max_rows=10)
                _save_csv(exp_df, out_dir / f"option_expiries_{code0.replace('.', '_')}.csv")
                metrics["option_expiries"] = {"ok": True, "rows": getattr(exp_df, "shape", [None])[0]}
            else:
                print(f"\n== option_expiration_date ({code0}) ==\nerror={exp_err}")
                metrics["option_expiries"] = {"ok": False, "error": exp_err}

            if bool(args.dump_option_chain):
                start = str(args.chain_start).strip() or None
                end = str(args.chain_end).strip() or None
                ok_c, chain_df, chain_err = _fetch_option_chain(
                    quote_ctx,
                    futu,
                    code0,
                    start=start,
                    end=end,
                    chain_right=str(args.chain_right),
                    chain_cond=str(args.chain_cond),
                )
                if ok_c:
                    _print_preview(f"option_chain ({code0})", chain_df, max_rows=10)
                    suffix = "all"
                    if start or end:
                        suffix = f"{(start or 'none')}_{(end or 'none')}"
                    _save_csv(chain_df, out_dir / f"option_chain_{code0.replace('.', '_')}_{suffix}.csv")
                    metrics["option_chain"] = {"ok": True, "rows": getattr(chain_df, "shape", [None])[0]}
                else:
                    print(f"\n== option_chain ({code0}) ==\nerror={chain_err}")
                    metrics["option_chain"] = {"ok": False, "error": chain_err}

        # 2b) Optional: historical volatility estimate
        if bool(args.calc_sigma) and kline_df is not None:
            sigma_res = _compute_hist_sigma(
                kline_df,
                window=int(args.sigma_window),
                trading_days=int(args.trading_days),
            )
            metrics["hist_sigma"] = sigma_res
            if sigma_res.get("ok"):
                print(
                    f"\n== hist_sigma ==\n"
                    f"sigma_annualized={sigma_res.get('sigma_annualized'):.6f} "
                    f"(window={sigma_res.get('window')}, trading_days={sigma_res.get('trading_days')})"
                )
            else:
                print(f"\n== hist_sigma ==\nerror={sigma_res.get('error')}")

        # 2b-vanilla) Batch vanilla option pricing (theoretical + market quote)
        if bool(args.price_vanilla):
            s0_for_bsm = _safe_float(_extract_scalar(snap, "last_price")) if "snap" in locals() else None
            if s0_for_bsm is None:
                s0_for_bsm = _safe_float(_extract_scalar(snap, "prev_close_price")) if "snap" in locals() else None

            sigma_for_bsm = float(args.sigma or 0.0)
            if sigma_for_bsm <= 0 and isinstance(metrics.get("hist_sigma"), dict):
                hs = metrics["hist_sigma"]
                if hs.get("ok"):
                    sigma_for_bsm = float(hs.get("sigma_annualized") or 0.0)

            if s0_for_bsm is None:
                print("\n== vanilla_pricing ==\nerror=missing S0 (snapshot.last_price)")
                metrics["vanilla_pricing"] = {"ok": False, "error": "missing S0"}
            elif sigma_for_bsm <= 0:
                print("\n== vanilla_pricing ==\nerror=missing sigma (pass --sigma or run with --calc-sigma)")
                metrics["vanilla_pricing"] = {"ok": False, "error": "missing sigma"}
            else:
                start = str(args.chain_start).strip() or None
                end = str(args.chain_end).strip() or None
                ok_c, chain_df, chain_err = _fetch_option_chain(
                    quote_ctx,
                    futu,
                    code0,
                    start=start,
                    end=end,
                    chain_right=str(args.vanilla_right),
                    chain_cond=str(args.chain_cond),
                )
                if not ok_c:
                    print(f"\n== vanilla_pricing ==\nerror=option_chain_fetch_failed: {chain_err}")
                    metrics["vanilla_pricing"] = {"ok": False, "error": f"option_chain_fetch_failed: {chain_err}"}
                else:
                    option_codes, filtered_chain = _build_option_universe(
                        chain_df,
                        expiry=str(args.vanilla_expiry).strip(),
                        right=str(args.vanilla_right),
                        limit=int(args.vanilla_max),
                    )

                    ok_q, qdf, qerr = _fetch_option_quotes(quote_ctx, futu, option_codes)
                    if not ok_q:
                        print(f"\n== option_quotes ==\nerror={qerr}")
                        merged = filtered_chain
                    else:
                        _print_preview("option_quotes", qdf, max_rows=5)
                        _save_csv(qdf, out_dir / f"option_quotes_{code0.replace('.', '_')}.csv")
                        merged = _merge_chain_and_quotes(filtered_chain, qdf)

                    try:
                        import pandas as pd  # type: ignore

                        if isinstance(merged, pd.DataFrame) and not merged.empty:
                            dfp = merged.copy()
                            strike_col = "strike_price" if "strike_price" in dfp.columns else None
                            expiry_col = None
                            for c in ("strike_time", "expiry_date", "maturity_date"):
                                if c in dfp.columns:
                                    expiry_col = c
                                    break
                            right_col = None
                            for c in ("option_type", "type", "call_put", "option_side"):
                                if c in dfp.columns:
                                    right_col = c
                                    break

                            t_list: list[float | None] = []
                            bs_prices: list[float | None] = []
                            for _, row in dfp.iterrows():
                                k_val = _safe_float(row.get(strike_col)) if strike_col else None
                                exp_dt = _to_date(row.get(expiry_col)) if expiry_col else None
                                t = max(0.0, (exp_dt - date.today()).days / 365.0) if exp_dt is not None else None
                                t_list.append(t)
                                if k_val is None or t is None:
                                    bs_prices.append(None)
                                    continue
                                is_call = True
                                if right_col is not None:
                                    is_call = "CALL" in str(row.get(right_col)).upper()
                                bs_prices.append(
                                    _bs_vanilla_price(
                                        is_call=is_call,
                                        s=float(s0_for_bsm),
                                        k=float(k_val),
                                        t=float(t),
                                        r=float(args.r),
                                        q=float(args.q),
                                        sigma=float(sigma_for_bsm),
                                    )
                                )

                            dfp = dfp.assign(
                                S0=float(s0_for_bsm),
                                sigma=float(sigma_for_bsm),
                                T_years=t_list,
                                bs_price=bs_prices,
                            )
                            out_path = out_dir / f"vanilla_pricing_{code0.replace('.', '_')}.csv"
                            _save_csv(dfp, out_path)
                            _print_preview("vanilla_pricing", dfp, max_rows=10)
                            metrics["vanilla_pricing"] = {
                                "ok": True,
                                "rows": int(getattr(dfp, "shape", [0])[0]),
                                "S0": float(s0_for_bsm),
                                "sigma": float(sigma_for_bsm),
                                "outfile": str(out_path),
                            }
                        else:
                            print("\n== vanilla_pricing ==\nerror=empty merged dataframe")
                            metrics["vanilla_pricing"] = {"ok": False, "error": "empty merged dataframe"}
                    except Exception as e:
                        print(f"\n== vanilla_pricing ==\nerror={e}")
                        metrics["vanilla_pricing"] = {"ok": False, "error": str(e)}

        # 2c) Optional: implied vol from an option quote (best-effort)
        if bool(args.calc_iv):
            option_code = str(args.option_code or "").strip()
            expiry = _parse_yyyy_mm_dd(str(args.expiry))
            strike = float(args.strike or 0.0)
            is_call = str(args.right).upper() == "CALL"

            s0 = _safe_float(_extract_scalar(snap, "last_price")) if "snap" in locals() else None
            if s0 is None:
                s0 = _safe_float(_extract_scalar(snap, "prev_close_price")) if "snap" in locals() else None

            if not option_code:
                # Try discover a contract from option chain (vanilla options).
                start = str(args.chain_start).strip()
                end = str(args.chain_end).strip()
                if not start and expiry is not None:
                    start = expiry.isoformat()
                if not end and expiry is not None:
                    end = expiry.isoformat()

                ok_c, chain_df, chain_err = _fetch_option_chain(
                    quote_ctx,
                    futu,
                    code0,
                    start=start or None,
                    end=end or None,
                    chain_right=str(args.right),
                    chain_cond=str(args.chain_cond),
                )
                if not ok_c:
                    print(f"\n== implied_vol ==\nerror=option_chain_fetch_failed: {chain_err}")
                    metrics["implied_vol"] = {"ok": False, "error": f"option_chain_fetch_failed: {chain_err}"}
                else:
                    pick = _pick_option_from_chain(
                        chain_df,
                        s0=s0,
                        strike=strike,
                        right=str(args.right),
                    )
                    if not pick.get("ok"):
                        print(f"\n== implied_vol ==\nerror=option_chain_pick_failed: {pick.get('error')}")
                        metrics["implied_vol"] = {
                            "ok": False,
                            "error": f"option_chain_pick_failed: {pick.get('error')}",
                        }
                    else:
                        option_code = str(pick.get("option_code") or "")
                        if strike <= 0 and pick.get("strike_price") is not None:
                            strike = float(pick["strike_price"])
                        if expiry is None and pick.get("expiry_date"):
                            expiry = _parse_yyyy_mm_dd(str(pick["expiry_date"]))
                        print(
                            f"\n== implied_vol (discovery) ==\n"
                            f"selected_option_code={option_code} selected_strike={strike} selected_expiry={expiry}"
                        )

            if not option_code:
                metrics.setdefault("implied_vol", {"ok": False, "error": "missing option_code"})
            else:
                ok, oq, err = _fetch_option_quote(quote_ctx, futu, option_code)
                if not ok:
                    print(f"\n== option_quote ({option_code}) ==\nerror={err}")
                    metrics["implied_vol"] = {"ok": False, "error": f"option quote fetch failed: {err}"}
                else:
                    _print_preview(f"option_quote ({option_code})", oq, max_rows=3)
                    _save_csv(oq, out_dir / f"option_quote_{option_code.replace('.', '_')}.csv")

                    bid = _safe_float(_extract_scalar(oq, "bid_price"))
                    ask = _safe_float(_extract_scalar(oq, "ask_price"))
                    last = _safe_float(_extract_scalar(oq, "last_price"))
                    mid = None
                    if bid is not None and ask is not None and ask >= bid:
                        mid = 0.5 * (bid + ask)
                    px = mid if bool(args.use_mid) and mid is not None else last or mid

                    if strike <= 0:
                        strike = _safe_float(_extract_scalar(oq, "strike_price")) or 0.0

                    if expiry is None:
                        for c in ("expiry_date", "maturity_date", "strike_time"):
                            v = _extract_scalar(oq, c)
                            expiry = _parse_yyyy_mm_dd(str(v)) if v is not None else None
                            if expiry is not None:
                                break

                    if s0 is None or strike <= 0 or expiry is None or px is None:
                        missing = []
                        if s0 is None:
                            missing.append("S0 (snapshot.last_price)")
                        if strike <= 0:
                            missing.append("K (pass --strike or quote strike_price)")
                        if expiry is None:
                            missing.append("expiry (pass --expiry or quote expiry_date)")
                        if px is None:
                            missing.append("option price (last or mid)")
                        msg = "missing: " + ", ".join(missing)
                        print(f"\n== implied_vol ==\nerror={msg}")
                        metrics["implied_vol"] = {"ok": False, "error": msg}
                    else:
                        t = max(0.0, (expiry - date.today()).days / 365.0)
                        iv = _implied_vol_bisect(
                            is_call=is_call,
                            s=float(s0),
                            k=float(strike),
                            t=float(t),
                            r=float(args.r),
                            q=float(args.q),
                            price=float(px),
                        )
                        if iv is None:
                            print("\n== implied_vol ==\nerror=failed to solve IV (check bounds/inputs)")
                            metrics["implied_vol"] = {
                                "ok": False,
                                "error": "iv_solve_failed",
                                "inputs": {
                                    "option_code": option_code,
                                    "S0": float(s0),
                                    "K": float(strike),
                                    "expiry": expiry.isoformat(),
                                    "T_years": float(t),
                                    "r": float(args.r),
                                    "q": float(args.q),
                                    "price": float(px),
                                    "use_mid": bool(args.use_mid),
                                    "right": "CALL" if is_call else "PUT",
                                },
                            }
                        else:
                            print(f"\n== implied_vol ==\niv={iv:.6f} (T={t:.6f}y, price={px})")
                            metrics["implied_vol"] = {
                                "ok": True,
                                "iv": float(iv),
                                "inputs": {
                                    "option_code": option_code,
                                    "S0": float(s0),
                                    "K": float(strike),
                                    "expiry": expiry.isoformat(),
                                    "T_years": float(t),
                                    "r": float(args.r),
                                    "q": float(args.q),
                                    "price": float(px),
                                    "bid": bid,
                                    "ask": ask,
                                    "last": last,
                                    "use_mid": bool(args.use_mid),
                                    "right": "CALL" if is_call else "PUT",
                                },
                            }

        # Save metrics (best-effort)
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = out_dir / "metrics.json"
            metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"\nsaved: {metrics_path}")
        except Exception:
            pass

        # 3) Optional real-time subscription (requires OpenD running & account permissions)
        if bool(args.subscribe_rt):
            try:
                # Avoid importing handler base classes at module import time when futu isn't installed.
                class _RTDataHandler(futu.RTDataHandlerBase):  # type: ignore
                    def on_recv_rsp(self, rsp_pb):  # type: ignore[override]
                        ret_code, data = super().on_recv_rsp(rsp_pb)
                        if ret_code != futu.RET_OK:
                            print(f"rt_push error: {data}")
                            return futu.RET_ERROR, data
                        print("rt_push:", data)
                        return futu.RET_OK, data

                quote_ctx.set_handler(_RTDataHandler())
                ret, sub = quote_ctx.subscribe([codes[0]], [futu.SubType.RT_DATA], session=futu.Session.ALL)
                if ret != futu.RET_OK:
                    print(f"subscribe error: {sub}")
                else:
                    print(sub)
                    time.sleep(max(1, int(args.rt_seconds)))
            except Exception as e:
                print(f"subscribe exception: {e}")

        return 0
    except Exception as e:
        msg = str(e)
        print(f"fatal: {msg}")
        print(
            "\nTips:\n"
            "- Install futu-api: `pip install futu-api`\n"
            "- Start Futu OpenD and keep it logged in\n"
            "- Ensure host/port match OpenD settings (default 127.0.0.1:11111)\n"
            "- For HK quotes, try code like HK.00700 (Tencent)\n"
        )
        return 2
    finally:
        if quote_ctx is not None:
            try:
                quote_ctx.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
 
