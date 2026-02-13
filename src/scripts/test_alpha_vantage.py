#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path


def _load_dotenv(dotenv_path: Path) -> None:
    """Minimal .env loader (no external deps, does not override env)."""
    if not dotenv_path.exists() or not dotenv_path.is_file():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test Alpha Vantage downloads with basic rate-limit retries")
    p.add_argument(
        "--symbols",
        default="JPM",
        help="Comma-separated equity symbols (default: JPM)",
    )
    p.add_argument("--start", default="2025-12-01", help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", default="2026-02-12", help="End date (YYYY-MM-DD)")
    p.add_argument(
        "--api-key",
        default=os.getenv("ALPHAVANTAGE_API_KEY", ""),
        help="Alpha Vantage API key (or set env ALPHAVANTAGE_API_KEY; loaded from .env if present)",
    )
    p.add_argument(
        "--outputsize",
        choices=("compact", "full"),
        default="compact",
        help="Alpha Vantage outputsize (default: compact)",
    )
    p.add_argument("--retries", type=int, default=6, help="Max retries (default: 6)")
    p.add_argument("--backoff", type=float, default=2.0, help="Base backoff seconds (default: 2.0)")
    p.add_argument("--max-sleep", type=float, default=60.0, help="Max sleep seconds (default: 60.0)")
    return p.parse_args()


def _looks_rate_limited(msg: str) -> bool:
    m = (msg or "").lower()
    return any(
        s in m
        for s in (
            "thank you for using alpha vantage",
            "our standard api rate limit",
            "frequency",
            "rate limit",
            "too many requests",
        )
    )


def safe_fetch_symbol(
    symbol: str,
    *,
    start: str,
    end: str,
    api_key: str,
    outputsize: str,
    retries: int,
    backoff: float,
    max_sleep: float,
) -> tuple[bool, "object", str | None, int]:
    try:
        from quant_research.data.sources.alpha_vantage import fetch_alpha_vantage_daily_adjusted
    except ImportError:  # allow direct execution without module context
        import sys

        sys.path.append(str(Path(__file__).resolve().parents[1]))
        from quant_research.data.sources.alpha_vantage import fetch_alpha_vantage_daily_adjusted

    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing dependency: pandas. Install it with `pip install pandas`.") from e

    last_err: str | None = None
    for attempt in range(1, retries + 1):
        try:
            df = fetch_alpha_vantage_daily_adjusted(
                symbol,
                start,
                end,
                api_key=api_key,
                outputsize=outputsize,
            )
            if df is None:
                df = pd.DataFrame()
            return True, df, None, attempt
        except Exception as e:
            last_err = str(e)
            if _looks_rate_limited(last_err):
                sleep_s = min(max_sleep, backoff * (2 ** (attempt - 1)) + random.uniform(0, backoff))
                print(f"[{symbol}] rate limited; retry in {sleep_s:.2f}s (attempt {attempt}/{retries})")
                time.sleep(sleep_s)
                continue
            return False, pd.DataFrame(), last_err, attempt
    return False, pd.DataFrame(), last_err, retries


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    _load_dotenv(project_root / ".env")

    args = _parse_args()
    api_key = (args.api_key or os.getenv("ALPHAVANTAGE_API_KEY") or "").strip()
    if not api_key:
        raise SystemExit(
            "Missing Alpha Vantage API key. Set ALPHAVANTAGE_API_KEY in .env or pass --api-key."
        )

    symbols = [s.strip() for s in str(args.symbols).split(",") if s.strip()]
    if not symbols:
        raise SystemExit("No symbols provided.")

    for sym in symbols:
        ok, df, err, attempts = safe_fetch_symbol(
            sym,
            start=str(args.start),
            end=str(args.end),
            api_key=api_key,
            outputsize=str(args.outputsize),
            retries=int(args.retries),
            backoff=float(args.backoff),
            max_sleep=float(args.max_sleep),
        )

        shape = getattr(df, "shape", None)
        cols = getattr(df, "columns", [])
        print(f"\n== {sym} ==")
        print(f"ok={ok} attempts={attempts} shape={shape}")
        if err:
            print(f"error={err}")
        print(f"columns={list(cols)}")
        try:
            print("head:")
            print(df.head(3))
            print("tail:")
            print(df.tail(3))
        except Exception:
            print("preview: <unavailable>")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

