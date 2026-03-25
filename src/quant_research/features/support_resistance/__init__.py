from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SupportResistanceConfig:
    """Configuration for support/resistance features."""

    window: int = 20
    touch_tolerance: float = 0.002
    close_buffer: float = 0.0005
    breakout_tolerance: float = 0.002
    volume_mult: float = 1.0
    channel_k: float = 2.0
    volume_zone_mult: float = 1.0


def _get_series(readouts: pd.DataFrame, *names: str) -> pd.Series:
    for name in names:
        if name in readouts.columns:
            return readouts[name]
    raise KeyError(f"Missing columns: {names}")


def _rolling_min(series: pd.Series, window: int) -> pd.Series:
    if isinstance(series.index, pd.MultiIndex):
        return series.groupby(level=1, group_keys=False).apply(lambda s: s.rolling(window).min())
    return series.rolling(window).min()


def _rolling_max(series: pd.Series, window: int) -> pd.Series:
    if isinstance(series.index, pd.MultiIndex):
        return series.groupby(level=1, group_keys=False).apply(lambda s: s.rolling(window).max())
    return series.rolling(window).max()


def _rolling_sum(series: pd.Series, window: int) -> pd.Series:
    if isinstance(series.index, pd.MultiIndex):
        return series.groupby(level=1, group_keys=False).apply(lambda s: s.rolling(window).sum())
    return series.rolling(window).sum()


def _rolling_mean(series: pd.Series, window: int) -> pd.Series:
    if isinstance(series.index, pd.MultiIndex):
        return series.groupby(level=1, group_keys=False).apply(lambda s: s.rolling(window).mean())
    return series.rolling(window).mean()


def _rolling_std(series: pd.Series, window: int) -> pd.Series:
    if isinstance(series.index, pd.MultiIndex):
        return series.groupby(level=1, group_keys=False).apply(lambda s: s.rolling(window).std())
    return series.rolling(window).std()


def _ewm_mean(series: pd.Series, span: int) -> pd.Series:
    if isinstance(series.index, pd.MultiIndex):
        return series.groupby(level=1, group_keys=False).apply(
            lambda s: s.ewm(span=span, adjust=False).mean()
        )
    return series.ewm(span=span, adjust=False).mean()


def _shift(series: pd.Series, periods: int) -> pd.Series:
    if isinstance(series.index, pd.MultiIndex):
        return series.groupby(level=1, group_keys=False).shift(periods)
    return series.shift(periods)


def _volume(readouts: pd.DataFrame) -> pd.Series:
    if "volume" in readouts.columns:
        return readouts["volume"]
    return pd.Series(1.0, index=readouts.index)


def _typical_price(readouts: pd.DataFrame) -> pd.Series:
    high = _get_series(readouts, "high", "High", "close", "Close")
    low = _get_series(readouts, "low", "Low", "close", "Close")
    close = _get_series(readouts, "close", "Close")
    return (high + low + close) / 3.0


def _bounce_long_signal(
    level: pd.Series,
    low: pd.Series,
    close: pd.Series,
    touch_tolerance: float,
    close_buffer: float,
) -> pd.Series:
    touch = low <= level * (1.0 + touch_tolerance)
    close_ok = close >= level * (1.0 + close_buffer)
    signal = np.where(touch & close_ok, 1.0, 0.0)
    signal = np.where(level.isna(), np.nan, signal)
    return pd.Series(signal, index=level.index)


def _breakout_long_signal(
    level: pd.Series, close: pd.Series, breakout_tolerance: float
) -> pd.Series:
    cond = close >= level * (1.0 + breakout_tolerance)
    signal = np.where(cond, 1.0, 0.0)
    signal = np.where(level.isna(), np.nan, signal)
    return pd.Series(signal, index=level.index)


def _shift_by_group(series: pd.Series, periods: int, groupby_level: Optional[int]) -> pd.Series:
    if periods == 0:
        return series
    if isinstance(series.index, pd.MultiIndex) and groupby_level is not None:
        return series.groupby(level=groupby_level, group_keys=False).shift(periods)
    return series.shift(periods)


def _poc_above_below_window(
    prices: np.ndarray,
    volumes: np.ndarray,
    window: int,
    bins: int,
    min_periods: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(prices)
    below = np.full(n, np.nan)
    above = np.full(n, np.nan)
    if window <= 0:
        return below, above

    for i in range(n):
        # Allow partial windows at the start of the series.
        start = max(0, i - window + 1)
        p_win = prices[start : i + 1]
        v_win = volumes[start : i + 1]
        mask = np.isfinite(p_win) & np.isfinite(v_win)
        if mask.sum() < min_periods:
            continue
        p_win = p_win[mask]
        v_win = v_win[mask]
        if p_win.size == 0:
            continue
        p_min = float(np.min(p_win))
        p_max = float(np.max(p_win))
        if not np.isfinite(p_min) or not np.isfinite(p_max):
            continue
        if p_min == p_max:
            below[i] = p_min
            above[i] = p_max
            continue

        edges = np.linspace(p_min, p_max, int(bins) + 1)
        idx = np.searchsorted(edges, p_win, side="right") - 1
        idx = np.clip(idx, 0, bins - 1)
        vol_bins = np.bincount(idx, weights=v_win, minlength=bins)
        centers = (edges[:-1] + edges[1:]) / 2.0

        current_price = prices[i]
        if not np.isfinite(current_price):
            continue

        below_mask = centers <= current_price
        above_mask = centers > current_price
        if below_mask.any():
            below_idx = int(np.argmax(np.where(below_mask, vol_bins, -1.0)))
            below[i] = centers[below_idx]
        if above_mask.any():
            above_idx = int(np.argmax(np.where(above_mask, vol_bins, -1.0)))
            above[i] = centers[above_idx]

    return below, above


def _poc_topk_above_below_window(
    prices: np.ndarray,
    volumes: np.ndarray,
    window: int,
    bins: int,
    min_periods: int,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rolling volume-profile top-k levels below/above current price.

    For each bar i, builds a volume histogram across price bins within the rolling window,
    then returns the top-k bin centers (ranked by volume desc, price asc) for:
    - below: centers <= current_price
    - above: centers > current_price
    """
    n = len(prices)
    k = int(k)
    if k <= 0:
        return np.full((n, 0), np.nan), np.full((n, 0), np.nan)
    below = np.full((n, k), np.nan)
    above = np.full((n, k), np.nan)
    if window <= 0:
        return below, above

    bins_i = max(int(bins), 2)

    for i in range(n):
        # Allow partial windows at the start of the series.
        start = max(0, i - window + 1)
        p_win = prices[start : i + 1]
        v_win = volumes[start : i + 1]
        mask = np.isfinite(p_win) & np.isfinite(v_win)
        if mask.sum() < min_periods:
            continue
        p_win = p_win[mask]
        v_win = v_win[mask]
        if p_win.size == 0:
            continue
        p_min = float(np.min(p_win))
        p_max = float(np.max(p_win))
        if not np.isfinite(p_min) or not np.isfinite(p_max):
            continue
        if p_min == p_max:
            below[i, 0] = p_min
            above[i, 0] = p_max
            continue

        edges = np.linspace(p_min, p_max, bins_i + 1)
        idx = np.searchsorted(edges, p_win, side="right") - 1
        idx = np.clip(idx, 0, bins_i - 1)
        vol_bins = np.bincount(idx, weights=v_win, minlength=bins_i)
        centers = (edges[:-1] + edges[1:]) / 2.0

        current_price = prices[i]
        if not np.isfinite(current_price):
            continue

        def _fill(mask_arr: np.ndarray, out_arr: np.ndarray) -> None:
            if not mask_arr.any():
                return
            cand_idx = np.flatnonzero(mask_arr)
            items = [(float(vol_bins[j]), float(centers[j])) for j in cand_idx]
            items.sort(key=lambda x: (-x[0], x[1]))  # volume desc, price asc
            for rank in range(min(k, len(items))):
                out_arr[rank] = items[rank][1]

        _fill(centers <= current_price, below[i])
        _fill(centers > current_price, above[i])

    return below, above


def volume_profile_poc_levels(
    readouts: pd.DataFrame,
    window: int = 240,
    bins: int = 40,
    price_col: Optional[str] = None,
    volume_col: str = "volume",
    min_periods: Optional[int] = None,
    shift: int = 1,
    groupby_level: Optional[int] = None,
) -> Tuple[pd.Series, pd.Series]:
    """POC above/below current price from rolling volume profile (minute-level)."""
    if price_col is None:
        price = _typical_price(readouts)
    else:
        price = readouts[price_col]
    volume = readouts[volume_col] if volume_col in readouts.columns else _volume(readouts)

    min_periods = int(min_periods) if min_periods is not None else int(window)
    min_periods = max(min_periods, 1)

    if isinstance(readouts.index, pd.MultiIndex) and groupby_level is not None:
        below = pd.Series(index=readouts.index, dtype=float)
        above = pd.Series(index=readouts.index, dtype=float)
        for _, grp in readouts.groupby(level=groupby_level, sort=False):
            p = price.loc[grp.index].to_numpy()
            v = volume.loc[grp.index].to_numpy()
            b, a = _poc_above_below_window(p, v, window, bins, min_periods)
            below.loc[grp.index] = b
            above.loc[grp.index] = a
    else:
        b, a = _poc_above_below_window(price.to_numpy(), volume.to_numpy(), window, bins, min_periods)
        below = pd.Series(b, index=readouts.index)
        above = pd.Series(a, index=readouts.index)

    below = _shift_by_group(below, shift, groupby_level)
    above = _shift_by_group(above, shift, groupby_level)
    return below, above


def volume_profile_poc_topk_levels(
    readouts: pd.DataFrame,
    window: int = 240,
    bins: int = 40,
    k: int = 3,
    price_col: Optional[str] = None,
    volume_col: str = "volume",
    min_periods: Optional[int] = None,
    shift: int = 1,
    groupby_level: Optional[int] = None,
) -> Tuple[list[pd.Series], list[pd.Series]]:
    """Top-k POC-like levels above/below current price from rolling volume profile.

    Returns two lists of length k: (below_levels, above_levels).
    Each element is a Series aligned to readouts.index.
    """
    if price_col is None:
        price = _typical_price(readouts)
    else:
        price = readouts[price_col]
    volume = readouts[volume_col] if volume_col in readouts.columns else _volume(readouts)

    min_periods = int(min_periods) if min_periods is not None else int(window)
    min_periods = max(min_periods, 1)
    k = int(k)
    if k <= 0:
        empty = [pd.Series(index=readouts.index, dtype=float) for _ in range(0)]
        return empty, empty

    def _split_cols(mat: np.ndarray, index: pd.Index) -> list[pd.Series]:
        return [pd.Series(mat[:, j], index=index) for j in range(mat.shape[1])]

    if isinstance(readouts.index, pd.MultiIndex) and groupby_level is not None:
        below_levels = [pd.Series(index=readouts.index, dtype=float) for _ in range(k)]
        above_levels = [pd.Series(index=readouts.index, dtype=float) for _ in range(k)]
        for _, grp in readouts.groupby(level=groupby_level, sort=False):
            p = price.loc[grp.index].to_numpy()
            v = volume.loc[grp.index].to_numpy()
            b, a = _poc_topk_above_below_window(p, v, window, bins, min_periods, k)
            b_cols = _split_cols(b, grp.index)
            a_cols = _split_cols(a, grp.index)
            for j in range(k):
                below_levels[j].loc[grp.index] = b_cols[j]
                above_levels[j].loc[grp.index] = a_cols[j]
    else:
        b, a = _poc_topk_above_below_window(price.to_numpy(), volume.to_numpy(), window, bins, min_periods, k)
        below_levels = _split_cols(b, readouts.index)
        above_levels = _split_cols(a, readouts.index)

    below_levels = [_shift_by_group(s, shift, groupby_level) for s in below_levels]
    above_levels = [_shift_by_group(s, shift, groupby_level) for s in above_levels]
    return below_levels, above_levels


def poc_bias_long(
    readouts: pd.DataFrame,
    window: int = 240,
    bins: int = 40,
    price_col: Optional[str] = None,
    volume_col: str = "volume",
    shift: int = 1,
    groupby_level: Optional[int] = None,
) -> pd.Series:
    """Long bias based on position between POC below/above (minute-level)."""
    close = _get_series(readouts, "close", "Close")
    poc_below, poc_above = volume_profile_poc_levels(
        readouts,
        window=window,
        bins=bins,
        price_col=price_col,
        volume_col=volume_col,
        min_periods=None,
        shift=shift,
        groupby_level=groupby_level,
    )
    width = poc_above - poc_below
    with np.errstate(divide="ignore", invalid="ignore"):
        bias = (poc_above - close) / width
    bias = np.clip(bias, 0.0, 1.0)
    bias = np.where(width <= 0, np.nan, bias)
    return pd.Series(bias, index=readouts.index)


def support_resistance_levels(readouts: pd.DataFrame, window: int) -> Tuple[pd.Series, pd.Series]:
    """Compute support/resistance levels from rolling low/high (shifted to avoid lookahead)."""
    low = _get_series(readouts, "low", "Low", "close", "Close")
    high = _get_series(readouts, "high", "High", "close", "Close")
    support = _shift(_rolling_min(low, window), 1)
    resistance = _shift(_rolling_max(high, window), 1)
    return support, resistance


def support_bounce_long(
    readouts: pd.DataFrame,
    window: int = 20,
    touch_tolerance: float = 0.002,
    close_buffer: float = 0.0005,
) -> pd.Series:
    """Long signal when price touches support and closes back above it."""
    support, _ = support_resistance_levels(readouts, window)
    low = _get_series(readouts, "low", "Low", "close", "Close")
    close = _get_series(readouts, "close", "Close")
    return _bounce_long_signal(support, low, close, touch_tolerance, close_buffer)


def resistance_breakout_long(
    readouts: pd.DataFrame,
    window: int = 20,
    breakout_tolerance: float = 0.002,
    volume_mult: float = 1.0,
) -> pd.Series:
    """Long signal when price breaks above resistance, optionally with volume confirmation."""
    _, resistance = support_resistance_levels(readouts, window)
    close = _get_series(readouts, "close", "Close")
    cond = close >= resistance * (1.0 + breakout_tolerance)

    if "volume" in readouts.columns and "adv20" in readouts.columns:
        vol = readouts["volume"]
        adv20 = readouts["adv20"]
        cond &= vol >= adv20 * float(volume_mult)

    signal = np.where(cond, 1.0, 0.0)
    signal = np.where(resistance.isna(), np.nan, signal)
    return pd.Series(signal, index=readouts.index)


def support_bias_long(readouts: pd.DataFrame, window: int = 20) -> pd.Series:
    """Continuous long bias: 1 near support, 0 near resistance."""
    support, resistance = support_resistance_levels(readouts, window)
    close = _get_series(readouts, "close", "Close")
    rng = resistance - support
    with np.errstate(divide="ignore", invalid="ignore"):
        position = (close - support) / rng
    bias = 1.0 - position
    bias = np.clip(bias, 0.0, 1.0)
    bias = np.where(rng <= 0, np.nan, bias)
    return pd.Series(bias, index=readouts.index)


def volume_cluster_levels(
    readouts: pd.DataFrame, window: int, zone_mult: float = 1.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """成交密集区: rolling volume-weighted mean + band."""
    price = _typical_price(readouts)
    volume = _volume(readouts)
    sum_w = _rolling_sum(volume, window)
    sum_wx = _rolling_sum(price * volume, window)
    sum_wx2 = _rolling_sum((price ** 2) * volume, window)

    with np.errstate(divide="ignore", invalid="ignore"):
        center_raw = sum_wx / sum_w
        var_raw = (sum_wx2 / sum_w) - (center_raw ** 2)
    var_raw = np.maximum(var_raw, 0.0)

    center = _shift(center_raw, 1)
    std = _shift(np.sqrt(var_raw), 1)
    zone_low = center - zone_mult * std
    zone_high = center + zone_mult * std
    return center, zone_low, zone_high


def volume_cluster_bounce_long(
    readouts: pd.DataFrame,
    window: int = 20,
    zone_mult: float = 1.0,
    touch_tolerance: float = 0.002,
    close_buffer: float = 0.0005,
) -> pd.Series:
    _, zone_low, _ = volume_cluster_levels(readouts, window, zone_mult=zone_mult)
    low = _get_series(readouts, "low", "Low", "close", "Close")
    close = _get_series(readouts, "close", "Close")
    return _bounce_long_signal(zone_low, low, close, touch_tolerance, close_buffer)


def moving_average_levels(
    readouts: pd.DataFrame, window: int
) -> Tuple[pd.Series, pd.Series]:
    close = _get_series(readouts, "close", "Close")
    sma = _shift(_rolling_mean(close, window), 1)
    ema = _shift(_ewm_mean(close, window), 1)
    return sma, ema


def rolling_vwap(readouts: pd.DataFrame, window: int) -> pd.Series:
    price = _typical_price(readouts)
    volume = _volume(readouts)
    sum_w = _rolling_sum(volume, window)
    sum_wx = _rolling_sum(price * volume, window)
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap_raw = sum_wx / sum_w
    return _shift(vwap_raw, 1)


def ma_support_long(
    readouts: pd.DataFrame,
    window: int = 20,
    touch_tolerance: float = 0.002,
    close_buffer: float = 0.0005,
) -> pd.Series:
    sma, _ = moving_average_levels(readouts, window)
    low = _get_series(readouts, "low", "Low", "close", "Close")
    close = _get_series(readouts, "close", "Close")
    return _bounce_long_signal(sma, low, close, touch_tolerance, close_buffer)


def vwap_support_long(
    readouts: pd.DataFrame,
    window: int = 20,
    touch_tolerance: float = 0.002,
    close_buffer: float = 0.0005,
) -> pd.Series:
    vwap = rolling_vwap(readouts, window)
    low = _get_series(readouts, "low", "Low", "close", "Close")
    close = _get_series(readouts, "close", "Close")
    return _bounce_long_signal(vwap, low, close, touch_tolerance, close_buffer)


def channel_levels(
    readouts: pd.DataFrame, window: int, channel_k: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    close = _get_series(readouts, "close", "Close")
    mid = _shift(_rolling_mean(close, window), 1)
    std = _shift(_rolling_std(close, window), 1)
    upper = mid + channel_k * std
    lower = mid - channel_k * std
    return mid, upper, lower


def channel_breakout_long(
    readouts: pd.DataFrame,
    window: int = 20,
    channel_k: float = 2.0,
    breakout_tolerance: float = 0.002,
) -> pd.Series:
    _, upper, _ = channel_levels(readouts, window, channel_k=channel_k)
    close = _get_series(readouts, "close", "Close")
    return _breakout_long_signal(upper, close, breakout_tolerance)


def channel_revert_long(
    readouts: pd.DataFrame,
    window: int = 20,
    channel_k: float = 2.0,
    touch_tolerance: float = 0.002,
    close_buffer: float = 0.0005,
) -> pd.Series:
    _, _, lower = channel_levels(readouts, window, channel_k=channel_k)
    low = _get_series(readouts, "low", "Low", "close", "Close")
    close = _get_series(readouts, "close", "Close")
    return _bounce_long_signal(lower, low, close, touch_tolerance, close_buffer)


def fibonacci_levels(
    readouts: pd.DataFrame, window: int
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    low = _get_series(readouts, "low", "Low", "close", "Close")
    high = _get_series(readouts, "high", "High", "close", "Close")
    roll_low = _shift(_rolling_min(low, window), 1)
    roll_high = _shift(_rolling_max(high, window), 1)
    rng = (roll_high - roll_low).where((roll_high - roll_low) > 0)
    lvl_236 = roll_high - rng * 0.236
    lvl_382 = roll_high - rng * 0.382
    lvl_500 = roll_high - rng * 0.500
    lvl_618 = roll_high - rng * 0.618
    return lvl_236, lvl_382, lvl_500, lvl_618


def fibonacci_bounce_long(
    readouts: pd.DataFrame,
    window: int = 20,
    touch_tolerance: float = 0.002,
    close_buffer: float = 0.0005,
) -> pd.Series:
    _, lvl_382, lvl_500, lvl_618 = fibonacci_levels(readouts, window)
    low = _get_series(readouts, "low", "Low", "close", "Close")
    close = _get_series(readouts, "close", "Close")
    sig_382 = _bounce_long_signal(lvl_382, low, close, touch_tolerance, close_buffer)
    sig_500 = _bounce_long_signal(lvl_500, low, close, touch_tolerance, close_buffer)
    sig_618 = _bounce_long_signal(lvl_618, low, close, touch_tolerance, close_buffer)
    return pd.concat([sig_382, sig_500, sig_618], axis=1).max(axis=1)


def fibonacci_swing_support(
    readouts: pd.DataFrame,
    lookaround_days: int = 3,
    min_high_mult: float = 1.3,
    min_high_points: float = 0.0,
    ratios: Iterable[float] = (0.236, 0.382, 0.5, 0.618),
    first_ratio: float = 0.236,
) -> Optional[dict[str, object]]:
    """Fibonacci retracement support levels from swing low to swing high.

    Definitions (as requested):
    - Low: within +/- `lookaround_days`, this day has the lowest low.
    - High: the max high after low, and meets thresholds:
        - If `min_high_mult` > 0: high >= low * min_high_mult
        - If `min_high_points` > 0: high - low >= min_high_points
    - Start drawing: only after price (low) falls to the first Fibonacci level.

    Notes:
    - This function is intended for daily bars (DatetimeIndex). It uses future
      data by definition (lookahead in pivot-low detection).
    - Returns None when no valid (low, high, touch) swing is found.
    """
    if readouts.empty:
        return None
    if isinstance(readouts.index, pd.MultiIndex):
        raise ValueError("fibonacci_swing_support expects daily bars with a DatetimeIndex.")

    lookaround_days = int(lookaround_days)
    if lookaround_days < 1:
        raise ValueError("lookaround_days must be >= 1.")

    df = readouts.copy()
    df = df.sort_index()
    low = _get_series(df, "low", "Low", "close", "Close").astype(float)
    high = _get_series(df, "high", "High", "close", "Close").astype(float)

    window = 2 * lookaround_days + 1
    roll_min = low.rolling(window=window, center=True).min()
    pivot_low_mask = roll_min.notna() & low.eq(roll_min)
    pivot_low_pos = np.flatnonzero(pivot_low_mask.to_numpy())
    if pivot_low_pos.size == 0:
        return None

    high_arr = high.to_numpy(dtype=float, copy=False)
    n = len(high_arr)
    future_max_high = np.full(n, np.nan, dtype=float)
    future_max_idx = np.full(n, -1, dtype=int)

    max_val = -np.inf
    max_idx = -1
    for i in range(n - 1, -1, -1):
        future_max_high[i] = max_val if max_idx >= 0 else np.nan
        future_max_idx[i] = max_idx
        val = high_arr[i]
        if np.isfinite(val) and val >= max_val:
            max_val = float(val)
            max_idx = int(i)

    first_ratio = float(first_ratio)
    ratio_vals = [float(r) for r in ratios]
    if first_ratio not in ratio_vals:
        ratio_vals = [first_ratio] + [r for r in ratio_vals if r != first_ratio]
    ratios_tuple = tuple(ratio_vals)

    low_arr = low.to_numpy(dtype=float, copy=False)
    touch_arr = low_arr  # touch condition uses low <= level
    best: Optional[dict[str, object]] = None

    for low_idx in pivot_low_pos.tolist():
        low_val = low_arr[low_idx]
        if not np.isfinite(low_val):
            continue
        high_idx = int(future_max_idx[low_idx])
        if high_idx <= low_idx:
            continue
        high_val = float(future_max_high[low_idx])
        if not np.isfinite(high_val):
            continue
        if float(min_high_mult) > 0 and high_val < float(min_high_mult) * float(low_val):
            continue
        if float(min_high_points) > 0 and (high_val - float(low_val)) < float(min_high_points):
            continue
        rng = high_val - float(low_val)
        if not np.isfinite(rng) or rng <= 0:
            continue

        lvl_first = high_val - rng * first_ratio
        after = touch_arr[high_idx:]
        if after.size == 0:
            continue
        cond = np.isfinite(after) & (after <= lvl_first)
        if not cond.any():
            continue
        touch_offset = int(np.argmax(cond))
        touch_idx = int(high_idx + touch_offset)

        candidate = {
            "low_idx": int(low_idx),
            "high_idx": int(high_idx),
            "touch_idx": int(touch_idx),
            "low": float(low_val),
            "high": float(high_val),
        }
        if best is None:
            best = candidate
            continue
        if touch_idx > int(best["touch_idx"]):  # type: ignore[arg-type]
            best = candidate
            continue
        if touch_idx == int(best["touch_idx"]) and high_idx > int(best["high_idx"]):  # type: ignore[arg-type]
            best = candidate

    if best is None:
        return None

    idx = low.index
    low_idx = int(best["low_idx"])  # type: ignore[arg-type]
    high_idx = int(best["high_idx"])  # type: ignore[arg-type]
    touch_idx = int(best["touch_idx"])  # type: ignore[arg-type]
    low_val = float(best["low"])  # type: ignore[arg-type]
    high_val = float(best["high"])  # type: ignore[arg-type]
    rng = high_val - low_val

    levels = {r: float(high_val - rng * r) for r in ratios_tuple}
    return {
        "low_date": pd.to_datetime(idx[low_idx]),
        "high_date": pd.to_datetime(idx[high_idx]),
        "touch_date": pd.to_datetime(idx[touch_idx]),
        "low": low_val,
        "high": high_val,
        "ratios": ratios_tuple,
        "first_ratio": first_ratio,
        "levels": levels,
    }


def build_support_resistance_features(
    readouts: pd.DataFrame,
    windows: Iterable[int] = (20,),
    config: SupportResistanceConfig | None = None,
) -> pd.DataFrame:
    """Append support/resistance features for the provided windows."""
    out = readouts.copy()
    cfg = config or SupportResistanceConfig()

    for window in windows:
        window = int(window)
        support, resistance = support_resistance_levels(out, window)
        out[f"sr_support_{window}"] = support
        out[f"sr_resistance_{window}"] = resistance
        out[f"sr_support_bounce_long_{window}"] = support_bounce_long(
            out,
            window=window,
            touch_tolerance=cfg.touch_tolerance,
            close_buffer=cfg.close_buffer,
        )
        out[f"sr_resistance_breakout_long_{window}"] = resistance_breakout_long(
            out,
            window=window,
            breakout_tolerance=cfg.breakout_tolerance,
            volume_mult=cfg.volume_mult,
        )
        out[f"sr_support_bias_long_{window}"] = support_bias_long(out, window=window)

        center, zone_low, zone_high = volume_cluster_levels(
            out, window=window, zone_mult=cfg.volume_zone_mult
        )
        out[f"sr_volume_node_{window}"] = center
        out[f"sr_volume_zone_low_{window}"] = zone_low
        out[f"sr_volume_zone_high_{window}"] = zone_high
        out[f"sr_volume_zone_bounce_long_{window}"] = volume_cluster_bounce_long(
            out,
            window=window,
            zone_mult=cfg.volume_zone_mult,
            touch_tolerance=cfg.touch_tolerance,
            close_buffer=cfg.close_buffer,
        )

        sma, ema = moving_average_levels(out, window)
        out[f"sr_sma_{window}"] = sma
        out[f"sr_ema_{window}"] = ema
        out[f"sr_vwap_{window}"] = rolling_vwap(out, window)
        out[f"sr_sma_support_long_{window}"] = ma_support_long(
            out,
            window=window,
            touch_tolerance=cfg.touch_tolerance,
            close_buffer=cfg.close_buffer,
        )
        out[f"sr_vwap_support_long_{window}"] = vwap_support_long(
            out,
            window=window,
            touch_tolerance=cfg.touch_tolerance,
            close_buffer=cfg.close_buffer,
        )

        mid, upper, lower = channel_levels(out, window, channel_k=cfg.channel_k)
        out[f"sr_channel_mid_{window}"] = mid
        out[f"sr_channel_upper_{window}"] = upper
        out[f"sr_channel_lower_{window}"] = lower
        out[f"sr_channel_breakout_long_{window}"] = channel_breakout_long(
            out,
            window=window,
            channel_k=cfg.channel_k,
            breakout_tolerance=cfg.breakout_tolerance,
        )
        out[f"sr_channel_revert_long_{window}"] = channel_revert_long(
            out,
            window=window,
            channel_k=cfg.channel_k,
            touch_tolerance=cfg.touch_tolerance,
            close_buffer=cfg.close_buffer,
        )

        fib_236, fib_382, fib_500, fib_618 = fibonacci_levels(out, window)
        out[f"sr_fib_236_{window}"] = fib_236
        out[f"sr_fib_382_{window}"] = fib_382
        out[f"sr_fib_500_{window}"] = fib_500
        out[f"sr_fib_618_{window}"] = fib_618
        out[f"sr_fib_bounce_long_{window}"] = fibonacci_bounce_long(
            out,
            window=window,
            touch_tolerance=cfg.touch_tolerance,
            close_buffer=cfg.close_buffer,
        )

    return out
