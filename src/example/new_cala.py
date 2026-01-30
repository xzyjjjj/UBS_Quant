#!/usr/bin/env python3
"""Structured example rewritten from assets/new_cala.ipynb.

Pipeline outline:
1) 数据准备
2) 环境初始化
3) 模型构建
4) 数据集构建
5) 回测配置
6) 数据示例
7) 训练记录
8) 信号生成
9) 信号分析
10) 组合回测
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Config:
    data_path: Path
    hdf_key: str
    target_col: str
    output_dir: Path
    t_exec: int
    exec_grid: Tuple[int, ...]


def data_preparation(cfg: Config) -> pd.DataFrame:
    """数据准备: load HDF and select target column."""
    raw = pd.read_hdf(cfg.data_path, key=cfg.hdf_key)
    if cfg.target_col not in raw.columns:
        raise KeyError(f"Missing target col {cfg.target_col!r} in HDF data.")
    return raw[cfg.target_col]


def environment_init() -> None:
    """环境初始化: set seeds and pandas options."""
    np.random.seed(42)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)


def _get_dates(target_raw: pd.DataFrame) -> List[pd.Timestamp]:
    if isinstance(target_raw.index, pd.MultiIndex):
        return list(target_raw.index.get_level_values(0).unique())
    return list(pd.Index(target_raw.index).unique())


def _slice_day(target_raw: pd.DataFrame, day) -> pd.DataFrame:
    return target_raw.loc[day]


def model_building(target_raw: pd.DataFrame) -> pd.DataFrame:
    """模型构建: compute daily snapshot and default signal."""
    dates = _get_dates(target_raw)
    readouts = pd.DataFrame(index=dates)

    last_close = []
    vwap = []
    for day in dates:
        day_df = _slice_day(target_raw, day)
        last_close.append(day_df["Close"].iloc[-1])
        vwap.append((day_df["Close"] * day_df["Volume"]).sum() / day_df["Volume"].sum())

    readouts["T_last_close"] = last_close
    readouts["T_vwap"] = vwap
    ratio = np.array(readouts["T_last_close"]) / np.array(readouts["T_vwap"])
    readouts["default_st"] = np.where(ratio > 1.0, -1, 1)
    return readouts


def dataset_building(target_raw: pd.DataFrame, readouts: pd.DataFrame) -> pd.DataFrame:
    """数据集构建: append strategy returns container."""
    readouts = readouts.copy()
    readouts["date"] = readouts.index
    return readouts


def backtest_config(cfg: Config) -> Iterable[int]:
    """回测配置: define execution time grid."""
    return cfg.exec_grid


def _price_at(day_df: pd.DataFrame, t_exec: int) -> float:
    if t_exec < len(day_df):
        return float(day_df["Close"].iloc[t_exec])
    return float("nan")


def _evaluate_return_t_p1(
    target_raw: pd.DataFrame, readouts: pd.DataFrame, t_exec: int, strategy_col: str
) -> pd.Series:
    dates = list(readouts.index)
    strategy = np.array(readouts[strategy_col].tolist()[:-2])

    short_price_t_p1 = []
    long_price_t_p2 = []
    for day in dates[1:-1]:
        short_price_t_p1.append(_price_at(_slice_day(target_raw, day), t_exec))
    for day in dates[2:]:
        long_price_t_p2.append(_price_at(_slice_day(target_raw, day), t_exec))

    short_price_t_p1 = np.array(short_price_t_p1)
    long_price_t_p2 = np.array(long_price_t_p2)
    return_rate = ((long_price_t_p2 / short_price_t_p1) - 1.0) * strategy
    return pd.Series(return_rate, index=dates[:-2])


def data_sample(readouts: pd.DataFrame, output_dir: Path) -> None:
    """数据示例: dump a small sample."""
    sample_path = output_dir / "data_sample.csv"
    readouts.head(10).to_csv(sample_path, index=True)


def training_record(sharpe_by_exec: pd.DataFrame, output_dir: Path) -> None:
    """训练记录: persist sharpe summary."""
    sharpe_path = output_dir / "sharpe_by_exec.csv"
    sharpe_by_exec.to_csv(sharpe_path, index=False)


def signal_generation(readouts: pd.DataFrame, output_dir: Path) -> None:
    """信号生成: export daily signal."""
    signal_path = output_dir / "signals.csv"
    readouts[["default_st"]].to_csv(signal_path, index=True)


def signal_analysis(sharpe_by_exec: pd.DataFrame, output_dir: Path) -> None:
    """信号分析: plot time sensitivity."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(sharpe_by_exec["t_exec"], sharpe_by_exec["sharpe"], c="blue", label="sharpe ratio")
    ax.scatter(sharpe_by_exec["t_exec"], sharpe_by_exec["sharpe"], c="blue")
    ax.set_title("time sensitivity of default_st")
    ax.set_xlabel("trading time (min)")
    ax.set_ylabel("sharpe ratio")
    ax.legend()
    fig.tight_layout()
    fig_path = output_dir / "time_sensitivity_default_st.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def portfolio_backtest(return_series: pd.Series, output_dir: Path) -> None:
    """组合回测: build equity curve."""
    equity = (1.0 + return_series.fillna(0.0)).cumprod()
    equity_path = output_dir / "equity_curve.csv"
    equity.to_csv(equity_path, index=True)


def main() -> int:
    cfg = Config(
        data_path=Path("assets/MinutesIdx.h5"),
        hdf_key="/d",
        target_col="IC",
        output_dir=Path("output/example"),
        t_exec=59,
        exec_grid=(0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 239),
    )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    environment_init()

    # 数据准备
    target_raw = data_preparation(cfg)

    # 模型构建
    readouts = model_building(target_raw)

    # 数据集构建
    readouts = dataset_building(target_raw, readouts)

    # 回测配置
    exec_grid = backtest_config(cfg)

    # 数据示例
    data_sample(readouts, cfg.output_dir)

    # 训练记录: compute sharpe across exec grid
    sharpe_rows = []
    for t_exec in exec_grid:
        returns = _evaluate_return_t_p1(target_raw, readouts, t_exec, "default_st")
        readouts[f"return_default_st_{t_exec}"] = returns
        ret_arr = returns.dropna().to_numpy()
        if ret_arr.size == 0 or np.std(ret_arr) == 0:
            sharpe = np.nan
        else:
            sharpe = np.sqrt(250.0) * np.mean(ret_arr) / np.std(ret_arr)
        sharpe_rows.append({"t_exec": t_exec, "sharpe": sharpe})

    sharpe_by_exec = pd.DataFrame(sharpe_rows)
    training_record(sharpe_by_exec, cfg.output_dir)

    # 信号生成
    signal_generation(readouts, cfg.output_dir)

    # 信号分析
    signal_analysis(sharpe_by_exec, cfg.output_dir)

    # 组合回测: use cfg.t_exec for example
    return_series = readouts[f"return_default_st_{cfg.t_exec}"]
    portfolio_backtest(return_series, cfg.output_dir)

    # 输出全量读数
    readouts.to_csv(cfg.output_dir / "readouts_full.csv", index=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
