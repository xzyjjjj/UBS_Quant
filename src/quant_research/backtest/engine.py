"""Backtest engine utilities."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def get_dates(target_raw: pd.DataFrame) -> List[pd.Timestamp]:
    if isinstance(target_raw.index, pd.MultiIndex):
        return list(target_raw.index.get_level_values(0).unique())
    return list(pd.Index(target_raw.index).unique())


def slice_day(target_raw: pd.DataFrame, day) -> pd.DataFrame:
    return target_raw.loc[day]


def _price_at(day_df: pd.DataFrame, t_exec: int) -> float:
    if t_exec < len(day_df):
        return float(day_df["Close"].iloc[t_exec])
    return float("nan")


def evaluate_return_t_p1(
    target_raw: pd.DataFrame, readouts: pd.DataFrame, t_exec: int, strategy_col: str
) -> pd.Series:
    dates = list(readouts.index)
    strategy = np.array(readouts[strategy_col].tolist()[:-2])

    short_price_t_p1 = []
    long_price_t_p2 = []
    for day in dates[1:-1]:
        short_price_t_p1.append(_price_at(slice_day(target_raw, day), t_exec))
    for day in dates[2:]:
        long_price_t_p2.append(_price_at(slice_day(target_raw, day), t_exec))

    short_price_t_p1 = np.array(short_price_t_p1)
    long_price_t_p2 = np.array(long_price_t_p2)
    return_rate = ((long_price_t_p2 / short_price_t_p1) - 1.0) * strategy
    return pd.Series(return_rate, index=dates[:-2])
