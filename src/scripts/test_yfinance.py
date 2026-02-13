#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test yfinance downloads with basic rate-limit retries")
    p.add_argument(
        "--tickers",
        default="JPM,^VIX,^TNX",
        help="Comma-separated tickers (default: JPM,^VIX,^TNX)",
    )
    p.add_argument("--period", default="1y", help="yfinance period (default: 1y)")
    p.add_argument("--interval", default="1d", help="yfinance interval (default: 1d)")
    p.add_argument("--retries", type=int, default=5, help="Max retries on rate limit (default: 5)")
    p.add_argument(
        "--backoff",
        type=float,
        default=1.0,
        help="Base backoff seconds (default: 1.0)",
    )
    p.add_argument(
        "--max-sleep",
        type=float,
        default=30.0,
        help="Max sleep between retries (default: 30.0)",
    )
    p.add_argument(
        "--retry-on-empty",
        action="store_true",
        help="Retry when download returns an empty DataFrame",
    )
    return p.parse_args()


def _is_rate_limit_error(exc: BaseException) -> bool:
    # yfinance has a dedicated exception in newer versions.
    if exc.__class__.__name__ == "YFRateLimitError":
        return True

    # requests HTTPError or similar.
    resp = getattr(exc, "response", None)
    status_code = getattr(resp, "status_code", None)
    if status_code == 429:
        return True

    msg = str(exc).lower()
    return any(s in msg for s in ("rate limit", "rate limited", "too many requests", "http 429", "429"))


@dataclass(frozen=True)
class DownloadResult:
    ok: bool
    df: "object"  # pandas.DataFrame at runtime
    error: str | None = None
    attempts: int = 0


def safe_download(
    ticker: str,
    *,
    period: str,
    interval: str,
    retries: int,
    backoff: float,
    max_sleep: float,
    retry_on_empty: bool,
) -> DownloadResult:
    try:
        import yfinance as yf  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing dependency: yfinance. Install it with `pip install yfinance`.") from e

    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing dependency: pandas. Install it with `pip install pandas`.") from e

    last_err: str | None = None

    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False,
                threads=False,
                auto_adjust=False,
            )

            if df is None:
                df = pd.DataFrame()

            if getattr(df, "empty", False) and retry_on_empty:
                last_err = "empty dataframe"
                sleep_s = min(max_sleep, backoff * (2 ** (attempt - 1)) + random.uniform(0, backoff))
                print(f"[{ticker}] empty dataframe; retry in {sleep_s:.2f}s (attempt {attempt}/{retries})")
                time.sleep(sleep_s)
                continue

            return DownloadResult(ok=True, df=df, error=None, attempts=attempt)
        except Exception as e:
            if _is_rate_limit_error(e):
                last_err = f"rate limited: {e}"
                sleep_s = min(max_sleep, backoff * (2 ** (attempt - 1)) + random.uniform(0, backoff))
                print(f"[{ticker}] rate limited; retry in {sleep_s:.2f}s (attempt {attempt}/{retries})")
                time.sleep(sleep_s)
                continue
            return DownloadResult(ok=False, df=pd.DataFrame(), error=str(e), attempts=attempt)

    return DownloadResult(ok=False, df=pd.DataFrame(), error=last_err, attempts=retries)


def _flatten_columns(df: "object") -> None:
    # In-place best-effort flatten for MultiIndex columns
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return

    if not isinstance(df, pd.DataFrame):
        return
    if not isinstance(df.columns, pd.MultiIndex):
        return
    df.columns = [c[0] if isinstance(c, tuple) and c else str(c) for c in df.columns]


def main() -> int:
    args = _parse_args()
    tickers = [t.strip() for t in str(args.tickers).split(",") if t.strip()]
    if not tickers:
        raise SystemExit("No tickers provided.")

    for t in tickers:
        res = safe_download(
            t,
            period=str(args.period),
            interval=str(args.interval),
            retries=int(args.retries),
            backoff=float(args.backoff),
            max_sleep=float(args.max_sleep),
            retry_on_empty=bool(args.retry_on_empty),
        )
        df = res.df
        _flatten_columns(df)

        shape = getattr(df, "shape", None)
        cols = getattr(df, "columns", [])
        idx = getattr(df, "index", [])
        print(f"\n== {t} ==")
        print(f"ok={res.ok} attempts={res.attempts} shape={shape}")
        if res.error:
            print(f"error={res.error}")
        print(f"columns={list(cols)[:12]}{' ...' if hasattr(cols, '__len__') and len(cols) > 12 else ''}")
        try:
            print("tail:")
            print(df.tail(3))
        except Exception:
            print("tail: <unavailable>")
        print(f"index_empty={len(idx) == 0 if hasattr(idx, '__len__') else 'unknown'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

