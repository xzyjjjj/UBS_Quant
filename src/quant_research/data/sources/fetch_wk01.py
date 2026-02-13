#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

try:
    from quant_research.data.sources.fred import DEFAULT_TREASURY_SERIES
    from quant_research.data.wk01 import (
        Week1Config,
        build_week1_panel,
        fetch_week1_raw,
        save_week1_outputs,
    )
except ImportError:  # allow direct execution without module context
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from quant_research.data.sources.fred import DEFAULT_TREASURY_SERIES
    from quant_research.data.wk01 import (
        Week1Config,
        build_week1_panel,
        fetch_week1_raw,
        save_week1_outputs,
    )


def _load_dotenv(dotenv_path: Path) -> None:
    """Minimal .env loader (no external deps)."""
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
    p = argparse.ArgumentParser(description="WK01: fetch JPM, VIX, and US Treasury yields")
    p.add_argument("--start", default="2018-01-01", help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", default="2024-12-31", help="End date (YYYY-MM-DD)")
    p.add_argument("--jpm", default="JPM", help="JPM ticker (default: JPM)")
    p.add_argument("--vix", default="^VIX", help="VIX ticker for Yahoo (default: ^VIX)")
    p.add_argument(
        "--vix-source",
        choices=("fred", "yahoo"),
        default="fred",
        help="VIX source (default: fred via VIXCLS)",
    )
    p.add_argument(
        "--vix-fred-series-id",
        default="VIXCLS",
        help="FRED series id for VIX when --vix-source=fred (default: VIXCLS)",
    )
    p.add_argument(
        "--treasury-series",
        default=",".join(DEFAULT_TREASURY_SERIES),
        help="Comma-separated FRED series ids (default: common DGS* set)",
    )
    p.add_argument(
        "--fred-api-key",
        default=os.getenv("FRED_API_KEY", ""),
        help="FRED API key (recommended; or set env FRED_API_KEY)",
    )
    p.add_argument(
        "--alpha-vantage-api-key",
        default=os.getenv("ALPHAVANTAGE_API_KEY", ""),
        help="Alpha Vantage API key (or set env ALPHAVANTAGE_API_KEY)",
    )
    p.add_argument(
        "--prefer-alpha-vantage-for-jpm",
        action="store_true",
        help="Use Alpha Vantage for JPM daily (otherwise use Yahoo/yfinance)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output directory (default: <project_root>/output/week1)",
    )
    return p.parse_args()


def main() -> int:
    project_root = Path(__file__).resolve().parents[4]
    _load_dotenv(project_root / ".env")

    args = _parse_args()
    out_dir = Path(args.out).expanduser().resolve() if args.out else (project_root / "output" / "week1")

    treasury_series = tuple(s.strip() for s in str(args.treasury_series).split(",") if s.strip())
    cfg = Week1Config(
        start=args.start,
        end=args.end,
        jpm_ticker=args.jpm,
        vix_ticker=args.vix,
        vix_source=str(args.vix_source),
        vix_fred_series_id=str(args.vix_fred_series_id),
        treasury_series=treasury_series or DEFAULT_TREASURY_SERIES,
        fred_api_key=args.fred_api_key or None,
        alpha_vantage_api_key=args.alpha_vantage_api_key or None,
        prefer_alpha_vantage_for_jpm=bool(args.prefer_alpha_vantage_for_jpm),
    )

    raw = fetch_week1_raw(cfg)
    panel = build_week1_panel(raw)
    save_week1_outputs(out_dir, raw, panel)

    print(f"Wrote: {out_dir}")
    for name in ("jpm", "vix", "treasury"):
        df = raw.get(name)
        shape = getattr(df, "shape", None)
        print(f"{name}: {shape}")
    if panel is not None and not panel.empty:
        print(f"Panel shape: {panel.shape}")
        print(f"Columns: {list(panel.columns)[:20]}{' ...' if len(panel.columns) > 20 else ''}")
    else:
        print("Panel is empty (fetch returned no data).")
        print("Common causes:")
        print("  - Yahoo rate limit: rerun later or use a different source for JPM")
        print("  - FRED 400: set FRED_API_KEY (env) or pass --fred-api-key")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
