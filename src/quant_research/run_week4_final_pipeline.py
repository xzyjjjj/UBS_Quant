#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run Week4 final submission pipeline with the selected final parameter set."
    )
    p.add_argument("--symbol", default="JPM", help="Underlying symbol (default: JPM)")
    p.add_argument(
        "--options-csv",
        default="output/market_panel/raw/options_history.csv",
        help="Raw options history CSV (default: output/market_panel/raw/options_history.csv)",
    )
    p.add_argument(
        "--panel-csv",
        default="output/market_panel/processed/panel.csv",
        help="Panel CSV (default: output/market_panel/processed/panel.csv)",
    )
    p.add_argument(
        "--equity-csv",
        default="output/market_panel/raw/equity_daily.csv",
        help="Equity daily CSV (default: output/market_panel/raw/equity_daily.csv)",
    )
    p.add_argument(
        "--prepared-csv",
        default="output/cme/jpm_options_final.csv",
        help="Prepared eval CSV path (default: output/cme/jpm_options_final.csv)",
    )
    p.add_argument(
        "--out-dir",
        default="output/week4_baseline_final",
        help="Week4 output directory (default: output/week4_baseline_final)",
    )
    p.add_argument(
        "--min-volume",
        type=float,
        default=10.0,
        help="Liquidity filter for baseline (default: 10)",
    )
    return p.parse_args()


def _run(cmd: list[str]) -> int:
    print("[RUN]", " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def main() -> int:
    args = _parse_args()
    py = sys.executable
    root = Path(__file__).resolve().parents[2]

    prepare_cmd = [
        py,
        str(root / "src" / "quant_research" / "run_week4_prepare_eval_data.py"),
        "--options-csv",
        str(args.options_csv),
        "--panel-csv",
        str(args.panel_csv),
        "--equity-csv",
        str(args.equity_csv),
        "--out-csv",
        str(args.prepared_csv),
        "--symbol",
        str(args.symbol),
        "--q-method",
        "trailing_div_12m",
        "--market-price-method",
        "mid_only",
        "--r-source",
        "DGS3MO",
        "--sigma-source",
        "blend_iv_rv252_75_25",
    ]
    rc = _run(prepare_cmd)
    if rc != 0:
        return rc

    baseline_cmd = [
        py,
        str(root / "src" / "quant_research" / "run_week4_baseline.py"),
        "--cme-csv",
        str(args.prepared_csv),
        "--panel-csv",
        str(args.panel_csv),
        "--sentiment-csv",
        str(root / "output" / "news" / "sentiment_daily.csv"),
        "--out-dir",
        str(args.out_dir),
        "--min-volume",
        str(float(args.min_volume)),
    ]
    rc = _run(baseline_cmd)
    if rc != 0:
        return rc

    print("[OK] Week4 final pipeline completed.")
    print("[INFO] final params:")
    print("  sigma_source=blend_iv_rv252_75_25")
    print("  r_source=DGS3MO")
    print("  q_method=trailing_div_12m")
    print("  market_price_method=mid_only")
    print(f"  min_volume={float(args.min_volume)}")
    print(f"  prepared_csv={args.prepared_csv}")
    print(f"  out_dir={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

