#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
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


def _parse_args(default_series: str) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test FRED downloads (VIX + Treasury yields)")
    p.add_argument("--start", default="2025-12-01", help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", default="2026-02-12", help="End date (YYYY-MM-DD)")
    p.add_argument(
        "--series",
        default=default_series,
        help="Comma-separated series ids (default: VIXCLS + common DGS*)",
    )
    p.add_argument(
        "--fred-api-key",
        default=os.getenv("FRED_API_KEY", ""),
        help="FRED API key (or set env FRED_API_KEY; loaded from .env if present)",
    )
    return p.parse_args()


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    _load_dotenv(project_root / ".env")

    try:
        from quant_research.data.sources.fred import DEFAULT_TREASURY_SERIES, fetch_fred_series_many
    except ImportError:  # allow direct execution without module context
        import sys

        sys.path.append(str(Path(__file__).resolve().parents[1]))
        from quant_research.data.sources.fred import DEFAULT_TREASURY_SERIES, fetch_fred_series_many

    default_series = ",".join(("VIXCLS", *DEFAULT_TREASURY_SERIES))
    args = _parse_args(default_series)

    series_ids = [s.strip() for s in str(args.series).split(",") if s.strip()]
    api_key = (args.fred_api_key or os.getenv("FRED_API_KEY") or "").strip() or None

    df = fetch_fred_series_many(series_ids, args.start, args.end, api_key=api_key)
    print(f"series={series_ids}")
    print(f"shape={df.shape}")
    print(f"columns={list(df.columns)}")
    print("head:")
    print(df.head(3))
    print("tail:")
    print(df.tail(3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
