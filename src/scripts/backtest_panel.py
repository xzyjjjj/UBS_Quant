#!/usr/bin/env python3
from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


def run_backtest(
    *,
    data_csv: Path,
    symbol: str,
    option_type: str,
    lookback_days: int,
    target_dte: int,
    dte_tol: int,
    max_strike_gap: float,
    contract_multiplier: float,
    force_daily_trade: bool,
    model_json: str,
    use_model_rank: bool,
    min_model_edge: float,
    min_volume: float,
    min_open_interest: float,
    max_moneyness_dev: float,
    out_csv: Path,
) -> tuple[bool, str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "backtest_10d_7d_near_spot.py"),
        "--data-csv",
        str(data_csv),
        "--symbol",
        str(symbol),
        "--option-type",
        str(option_type),
        "--lookback-days",
        str(int(lookback_days)),
        "--target-dte",
        str(int(target_dte)),
        "--dte-tol",
        str(int(dte_tol)),
        "--max-strike-gap",
        str(float(max_strike_gap)),
        "--contract-multiplier",
        str(float(contract_multiplier)),
        "--model-json",
        str(model_json),
        "--min-model-edge",
        str(float(min_model_edge)),
        "--min-volume",
        str(float(min_volume)),
        "--min-open-interest",
        str(float(min_open_interest)),
        "--max-moneyness-dev",
        str(float(max_moneyness_dev)),
        "--out-csv",
        str(out_csv),
    ]
    if force_daily_trade:
        cmd.append("--force-daily-trade")
    if use_model_rank:
        cmd.append("--use-model-rank")
    p = subprocess.run(cmd, capture_output=True, text=True)
    msg = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
    return p.returncode == 0, msg.strip()


def safe_float(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    return v if math.isfinite(v) else None


def compute_metrics(df: pd.DataFrame) -> dict[str, float]:
    executed = df[df["status"] == "EXECUTED"].copy()
    if executed.empty:
        return {
            "trades": 0.0,
            "total_pnl": 0.0,
            "avg_pnl": 0.0,
            "win_rate": 0.0,
            "max_drawdown": 0.0,
            "sharpe_daily": 0.0,
            "profit_factor": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "win_loss_ratio": 0.0,
        }
    pnl = pd.to_numeric(executed["pnl_1lot"], errors="coerce").fillna(0.0)
    cum = pnl.cumsum()
    running_max = cum.cummax()
    drawdown = cum - running_max
    wins = (pnl > 0).sum()
    losses = (pnl < 0).sum()
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = -pnl[pnl < 0].sum()
    avg_win = float(pnl[pnl > 0].mean()) if wins > 0 else 0.0
    avg_loss = float((-pnl[pnl < 0]).mean()) if losses > 0 else 0.0
    mu = pnl.mean()
    sd = pnl.std(ddof=1) if len(pnl) > 1 else 0.0
    sharpe = (mu / sd * math.sqrt(252.0)) if sd > 1e-12 else 0.0
    pf = (gross_profit / gross_loss) if gross_loss > 1e-12 else float("inf")
    return {
        "trades": float(len(executed)),
        "total_pnl": float(pnl.sum()),
        "avg_pnl": float(mu),
        "win_rate": float(wins / max(len(pnl), 1)),
        "max_drawdown": float(drawdown.min() if len(drawdown) else 0.0),
        "sharpe_daily": float(sharpe),
        "profit_factor": float(pf),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "win_loss_ratio": float(avg_win / avg_loss) if avg_loss > 1e-12 else float("inf"),
        "wins": float(wins),
        "losses": float(losses),
    }


def build_equity_curve(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["status"] == "EXECUTED"].copy()
    if d.empty:
        return pd.DataFrame(columns=["trade_date", "cum_pnl"])
    d["trade_date"] = pd.to_datetime(d["trade_date"], errors="coerce")
    d["pnl_1lot"] = pd.to_numeric(d["pnl_1lot"], errors="coerce").fillna(0.0)
    d = d.sort_values("trade_date")
    d["cum_pnl"] = d["pnl_1lot"].cumsum()
    return d[["trade_date", "cum_pnl"]]


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    st.set_page_config(page_title="JPM Backtest Panel", layout="wide")
    st.title("JPM Backtest Panel")

    with st.sidebar:
        st.header("Backtest Setup")
        data_csv = st.text_input(
            "Data CSV",
            value=str(root / "output" / "server_bundle_week5" / "data" / "processed" / "jpm_options_final.csv"),
        )
        symbol = st.text_input("Symbol", value="JPM")
        option_type = st.selectbox("Option Type", ["call", "put", "both"], index=0)
        lookback_days = st.number_input("Lookback Trade Days", min_value=10, max_value=1200, value=252, step=10)
        target_dte = st.number_input("Target DTE", min_value=1, max_value=90, value=14, step=1)
        dte_tol = st.number_input("DTE Tolerance", min_value=0, max_value=30, value=10, step=1)
        max_strike_gap = st.number_input("Max |Strike-Spot|", min_value=0.5, max_value=50.0, value=15.0, step=0.5)
        max_moneyness_dev = st.number_input("Max |S/K - 1|", min_value=0.01, max_value=1.0, value=0.30, step=0.01, format="%.2f")
        min_volume = st.number_input("Min Volume", min_value=0.0, max_value=10000000.0, value=1.0, step=10.0)
        min_open_interest = st.number_input("Min Open Interest", min_value=0.0, max_value=10000000.0, value=1.0, step=10.0)
        contract_multiplier = st.number_input("Contract Multiplier", min_value=1.0, max_value=1000.0, value=100.0, step=1.0)
        force_daily_trade = st.checkbox("Force Daily Trade", value=False)
        st.header("Model Strategy")
        model_json = st.text_input("Model JSON", value=str(root / "models" / "bsm_mlp_residual.json"))
        use_model_rank = st.checkbox("Use Model Edge Rank", value=True)
        min_model_edge = st.number_input("Min Model Edge", value=0.0, step=0.1, format="%.4f")
        out_csv = st.text_input(
            "Output CSV",
            value=str(root / "output" / "week6_backtest_panel_trades.csv"),
        )
        run_btn = st.button("Run Backtest")

    if not run_btn:
        st.info("Configure parameters in sidebar and click `Run Backtest`.")
        return

    ok, log = run_backtest(
        data_csv=Path(data_csv),
        symbol=symbol,
        option_type=option_type,
        lookback_days=int(lookback_days),
        target_dte=int(target_dte),
        dte_tol=int(dte_tol),
        max_strike_gap=float(max_strike_gap),
        contract_multiplier=float(contract_multiplier),
        force_daily_trade=bool(force_daily_trade),
        model_json=str(model_json),
        use_model_rank=bool(use_model_rank),
        min_model_edge=float(min_model_edge),
        min_volume=float(min_volume),
        min_open_interest=float(min_open_interest),
        max_moneyness_dev=float(max_moneyness_dev),
        out_csv=Path(out_csv),
    )
    st.subheader("Engine Log")
    st.code(log if log else "(empty)")
    if not ok:
        st.error("Backtest run failed.")
        return

    out_path = Path(out_csv)
    if not out_path.exists():
        st.error(f"Output file not found: {out_path}")
        return

    df = pd.read_csv(out_path)
    if df.empty:
        st.warning("Backtest output is empty.")
        return

    m = compute_metrics(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trades", f"{int(m['trades'])}")
    c2.metric("Total PnL", f"{m['total_pnl']:.2f}")
    c3.metric("Avg PnL/Trade", f"{m['avg_pnl']:.2f}")
    c4.metric("Win Rate", f"{100.0*m['win_rate']:.2f}%")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Max Drawdown", f"{m['max_drawdown']:.2f}")
    c6.metric("Sharpe (Daily)", f"{m['sharpe_daily']:.3f}")
    c7.metric("Profit Factor", "inf" if math.isinf(m["profit_factor"]) else f"{m['profit_factor']:.3f}")
    c8.metric("Wins / Losses", f"{int(m['wins'])}/{int(m['losses'])}")

    c9, c10, c11 = st.columns(3)
    c9.metric("Avg Win", f"{m['avg_win']:.2f}")
    c10.metric("Avg Loss", f"{m['avg_loss']:.2f}")
    c11.metric("Win/Loss Ratio", "inf" if math.isinf(m["win_loss_ratio"]) else f"{m['win_loss_ratio']:.3f}")

    st.subheader("Strategy Setup")
    st.markdown(
        f"- `model_json`: `{model_json}`\n"
        f"- `use_model_rank`: `{use_model_rank}`\n"
        f"- `min_model_edge`: `{min_model_edge}`\n"
        f"- `target_dte ± tol`: `{target_dte} ± {dte_tol}`\n"
        f"- `max_strike_gap`: `{max_strike_gap}`\n"
        f"- `max_moneyness_dev`: `{max_moneyness_dev}`\n"
        f"- `min_volume / min_open_interest`: `{min_volume} / {min_open_interest}`\n"
        f"- `force_daily_trade`: `{force_daily_trade}`"
    )

    st.subheader("Equity Curve")
    curve = build_equity_curve(df)
    if curve.empty:
        st.info("No executed trades for equity curve.")
    else:
        plot_df = curve.set_index("trade_date")
        st.line_chart(plot_df["cum_pnl"])

    st.subheader("All Actions")
    show_cols = [
        "trade_date",
        "status",
        "selection_mode",
        "model_edge",
        "model_fair",
        "contract_id",
        "option_type",
        "strike",
        "premium",
        "expiration",
        "spot_trade",
        "spot_expiry_used",
        "intrinsic_expiry",
        "pnl_per_contract",
        "pnl_1lot",
    ]
    cols = [c for c in show_cols if c in df.columns]
    st.dataframe(df[cols], use_container_width=True, hide_index=True)

    if "model_edge" in df.columns:
        st.subheader("Trade Process (Model Edge by Day)")
        proc = df.copy()
        proc["trade_date"] = pd.to_datetime(proc["trade_date"], errors="coerce")
        proc["model_edge"] = pd.to_numeric(proc["model_edge"], errors="coerce")
        proc = proc.sort_values("trade_date")
        st.line_chart(proc.set_index("trade_date")[["model_edge"]].fillna(0.0))
    st.download_button(
        "Download Actions CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=Path(out_csv).name,
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
