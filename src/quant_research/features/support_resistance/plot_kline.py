from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import json
import numpy as np
import pandas as pd

try:
    from quant_research.backtest.engine import get_dates, slice_day
    from quant_research.data.loader import load_target_series
    from quant_research.features.support_resistance import (
        fibonacci_swing_support,
        volume_cluster_levels,
        volume_profile_poc_levels,
        volume_profile_poc_topk_levels,
    )
except ImportError:  # allow direct execution without module context
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from quant_research.backtest.engine import get_dates, slice_day
    from quant_research.data.loader import load_target_series
    from quant_research.features.support_resistance import (
        fibonacci_swing_support,
        volume_cluster_levels,
        volume_profile_poc_levels,
        volume_profile_poc_topk_levels,
    )


@dataclass(frozen=True)
class Config:
    data_path: Path
    hdf_key: str
    target_col: str
    output_dir: Path
    start: Optional[str] = None
    end: Optional[str] = None
    max_days: int = 120
    max_bars: int = 2000
    filename: str = "kline.png"
    html_filename: str = "kline.html"
    daily_filename: str = "kline_daily.png"
    daily_html_filename: str = "kline_daily.html"
    interactive_renderer: str = "plotly"
    show_volume_zone: bool = False
    volume_zone_window: int = 20
    volume_zone_mult: float = 1.0
    show_top_sr: bool = True
    top_sr_n: int = 3
    show_poc: bool = True
    poc_window: int = 240
    poc_windows: Tuple[int, ...] = (240,)
    poc_bins: int = 40
    poc_shift: int = 1
    poc_groupby_level: Optional[int] = None
    show_daily_poc: bool = True
    daily_poc_window: int = 60
    daily_poc_windows: Tuple[int, ...] = (60, 120)
    daily_poc_bins: int = 40
    daily_poc_shift: int = 1
    poc_price_col: Optional[str] = None
    poc_volume_col: str = "volume"
    show_fib: bool = True
    fib_lookaround_days: int = 3
    fib_min_high_mult: float = 1.3
    fib_min_high_points: float = 0.0
    fib_ratios: Tuple[float, ...] = (0.236, 0.382, 0.5, 0.618)
    fib_first_ratio: float = 0.236
    fib_calc_days: int = 250
    use_minute_kline: bool = True
    export_daily_only: bool = True
    contract: Optional[str] = None
    price_col: Optional[str] = None
    price_cols: Tuple[str, ...] = (
        "Close",
        "close",
        "last",
        "last_price",
        "lastprice",
        "price",
        "Price",
        "T_last_close",
    )
    open_cols: Tuple[str, ...] = (
        "Open",
        "open",
        "O",
        "o",
        "open_price",
        "openprice",
        "OpenPrice",
        "openPrice",
        "T_open",
    )
    high_cols: Tuple[str, ...] = (
        "High",
        "high",
        "H",
        "h",
        "high_price",
        "highprice",
        "HighPrice",
        "highPrice",
        "T_high",
    )
    low_cols: Tuple[str, ...] = (
        "Low",
        "low",
        "L",
        "l",
        "low_price",
        "lowprice",
        "LowPrice",
        "lowPrice",
        "T_low",
    )
    close_cols: Tuple[str, ...] = (
        "Close",
        "close",
        "C",
        "c",
        "last",
        "last_price",
        "lastprice",
        "close_price",
        "closeprice",
        "ClosePrice",
        "closePrice",
        "T_close",
        "T_last_close",
    )
    volume_cols: Tuple[str, ...] = (
        "Volume",
        "volume",
        "VOL",
        "vol",
        "qty",
        "Qty",
    )


def _build_daily_ohlc(target_raw: pd.DataFrame) -> pd.DataFrame:
    dates = get_dates(target_raw)
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []
    for day in dates:
        day_df = slice_day(target_raw, day)
        close_series = day_df["Close"] if "Close" in day_df.columns else day_df["close"]
        if "Open" in day_df.columns:
            open_series = day_df["Open"]
        elif "open" in day_df.columns:
            open_series = day_df["open"]
        else:
            open_series = close_series
        if "High" in day_df.columns:
            high_series = day_df["High"]
        elif "high" in day_df.columns:
            high_series = day_df["high"]
        else:
            high_series = close_series
        if "Low" in day_df.columns:
            low_series = day_df["Low"]
        elif "low" in day_df.columns:
            low_series = day_df["low"]
        else:
            low_series = close_series
        if "Volume" in day_df.columns:
            volume_series = day_df["Volume"]
        elif "volume" in day_df.columns:
            volume_series = day_df["volume"]
        else:
            volume_series = None

        opens.append(float(open_series.iloc[0]))
        highs.append(float(high_series.max()))
        lows.append(float(low_series.min()))
        closes.append(float(close_series.iloc[-1]))
        if volume_series is None:
            volumes.append(float("nan"))
        else:
            volumes.append(float(volume_series.sum()))

    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        },
        index=pd.to_datetime(pd.Index(dates)),
    )
    return df


def _normalize_col_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _resolve_col(
    df: pd.DataFrame, candidates: Iterable[str], *, kind: str
) -> pd.Series:
    for name in candidates:
        if name in df.columns:
            return _coerce_series(df[name], kind=kind)

    normalized = {_normalize_col_name(col): col for col in df.columns}
    for name in candidates:
        key = _normalize_col_name(name)
        if key in normalized:
            return _coerce_series(df[normalized[key]], kind=kind)

    if isinstance(df.columns, pd.MultiIndex):
        for name in candidates:
            matches = []
            for col in df.columns:
                if any(name == str(part) for part in col):
                    matches.append(col)
                else:
                    for part in col:
                        if _normalize_col_name(name) == _normalize_col_name(str(part)):
                            matches.append(col)
                            break
            if matches:
                return _coerce_series(df.loc[:, matches], kind=kind)

    raise KeyError(
        f"Missing {kind} column. Tried: {tuple(candidates)}. Available: {list(df.columns)}"
    )


def _coerce_series(col: pd.Series | pd.DataFrame, *, kind: str) -> pd.Series:
    if isinstance(col, pd.Series):
        return col
    if isinstance(col, pd.DataFrame):
        if col.shape[1] == 1:
            return col.iloc[:, 0]
        raise KeyError(
            f"Multiple {kind} columns matched: {list(col.columns)}. "
            "Set Config.price_col or specify a unique column."
        )
    raise TypeError(f"Unexpected column type for {kind}: {type(col)}")


def _build_minute_ohlc(
    target_raw: pd.DataFrame,
    open_cols: Iterable[str],
    high_cols: Iterable[str],
    low_cols: Iterable[str],
    close_cols: Iterable[str],
    volume_cols: Iterable[str],
    price_col: Optional[str] = None,
    price_cols: Iterable[str] = (),
) -> pd.DataFrame:
    if not isinstance(target_raw, pd.DataFrame):
        raise ValueError("Minute-level OHLC requires target_raw as DataFrame.")

    try:
        open_ = _resolve_col(target_raw, open_cols, kind="open")
        high = _resolve_col(target_raw, high_cols, kind="high")
        low = _resolve_col(target_raw, low_cols, kind="low")
        close = _resolve_col(target_raw, close_cols, kind="close")
    except KeyError:
        price = None
        if price_col:
            try:
                price = _resolve_col(target_raw, (price_col,), kind="price")
            except KeyError:
                price = None
        if price is None:
            for name in price_cols:
                if name in target_raw.columns:
                    price = _coerce_series(target_raw[name], kind="price")
                    break
            if price is None:
                normalized = {_normalize_col_name(col): col for col in target_raw.columns}
                for name in price_cols:
                    key = _normalize_col_name(name)
                    if key in normalized:
                        price = _coerce_series(target_raw[normalized[key]], kind="price")
                        break
        if price is None:
            numeric_cols = [
                col for col in target_raw.columns if pd.api.types.is_numeric_dtype(target_raw[col])
            ]
            if len(numeric_cols) == 1:
                price = _coerce_series(target_raw[numeric_cols[0]], kind="price")
            else:
                raise KeyError(
                    "Missing OHLC columns and unable to infer price. "
                    f"Set Config.price_col to one of: {list(target_raw.columns)}"
                )

        price = _coerce_series(price, kind="price")
        open_ = price
        high = price
        low = price
        close = price

    open_ = _coerce_series(open_, kind="open")
    high = _coerce_series(high, kind="high")
    low = _coerce_series(low, kind="low")
    close = _coerce_series(close, kind="close")
    try:
        volume = _resolve_col(target_raw, volume_cols, kind="volume")
    except KeyError:
        volume = pd.Series(np.nan, index=target_raw.index)

    df = pd.DataFrame(
        {
            "open": open_.astype(float),
            "high": high.astype(float),
            "low": low.astype(float),
            "close": close.astype(float),
            "volume": volume.astype(float),
        },
        index=target_raw.index,
    )
    return df


def _daily_ohlc_from_minute(ohlc: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(ohlc.index, pd.MultiIndex):
        return ohlc
    grouped = ohlc.groupby(level=0, sort=False)
    daily = pd.DataFrame(
        {
            "open": grouped["open"].first(),
            "high": grouped["high"].max(),
            "low": grouped["low"].min(),
            "close": grouped["close"].last(),
            "volume": grouped["volume"].sum(),
        }
    )
    daily.index = pd.to_datetime(daily.index)
    return daily


def _load_raw_hdf(data_path: Path, hdf_key: str) -> pd.DataFrame:
    raw = pd.read_hdf(data_path, key=hdf_key)
    if not isinstance(raw, pd.DataFrame):
        raise ValueError("HDF data must be a DataFrame with OHLC columns.")
    return raw


def _unwrap_nested_frames(raw: pd.DataFrame, target_col: Optional[str]) -> pd.DataFrame:
    def _series_has_frame(series: pd.Series) -> bool:
        if series.empty:
            return False
        sample = series.dropna()
        if sample.empty:
            return False
        return isinstance(sample.iloc[0], pd.DataFrame)

    # Case 1: columns are MultiIndex with contract at level 0.
    if isinstance(raw.columns, pd.MultiIndex) and target_col:
        if target_col in raw.columns.get_level_values(0):
            return raw[target_col]

    series: Optional[pd.Series] = None
    if target_col and target_col in raw.columns:
        candidate = raw[target_col]
        if isinstance(candidate, pd.Series) and _series_has_frame(candidate):
            series = candidate
        elif isinstance(candidate, pd.DataFrame):
            return candidate
    if series is None and raw.shape[1] == 1 and _series_has_frame(raw.iloc[:, 0]):
        series = raw.iloc[:, 0]
    if series is None:
        candidates = []
        for col in raw.columns:
            col_series = raw[col] if isinstance(raw[col], pd.Series) else None
            if col_series is not None and _series_has_frame(col_series):
                candidates.append(col)
        if candidates:
            chosen = target_col if target_col in candidates else candidates[0]
            if target_col and target_col not in candidates:
                print(f"[warn] contract {target_col!r} not found, using {chosen!r} instead.")
            series = raw[chosen]
        else:
            return raw

    frames = []
    for idx, cell in series.items():
        if isinstance(cell, pd.DataFrame):
            frame = cell.copy()
            if "Date" not in frame.columns and "date" not in frame.columns:
                frame["Date"] = idx
            frames.append(frame)
    if not frames:
        return raw
    out = pd.concat(frames, ignore_index=True)
    return out


def _maybe_make_multiindex(raw: pd.DataFrame) -> pd.DataFrame:
    if isinstance(raw.index, (pd.MultiIndex, pd.DatetimeIndex)):
        return raw
    if "Date" not in raw.columns or "Time" not in raw.columns:
        return raw

    date_raw = raw["Date"]
    time_raw = raw["Time"]

    if pd.api.types.is_datetime64_any_dtype(date_raw):
        date_vals = pd.to_datetime(date_raw)
    else:
        date_vals = pd.to_datetime(date_raw.astype(str), format="%Y%m%d", errors="coerce")

    time_str = time_raw.astype(str).str.replace(r"[^0-9]", "", regex=True).str.zfill(4)
    hours = pd.to_numeric(time_str.str.slice(0, 2), errors="coerce").fillna(0).astype(int)
    minutes = pd.to_numeric(time_str.str.slice(2, 4), errors="coerce").fillna(0).astype(int)

    # Normalize invalid times and convert to minute-of-day.
    hours = hours.clip(lower=0, upper=23)
    minutes = minutes.clip(lower=0, upper=59)
    minute_of_day = hours * 60 + minutes

    idx = pd.MultiIndex.from_arrays([date_vals, minute_of_day], names=["date", "minute"])
    out = raw.copy()
    out.index = idx
    return out


def _apply_date_filter(df: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    out = df
    if start or end:
        if isinstance(out.index, pd.MultiIndex):
            dates = pd.to_datetime(out.index.get_level_values(0))
            mask = np.ones(len(out), dtype=bool)
            if start:
                mask &= dates >= pd.to_datetime(start)
            if end:
                mask &= dates <= pd.to_datetime(end)
            out = out[mask]
        else:
            if start:
                out = out[out.index >= pd.to_datetime(start)]
            if end:
                out = out[out.index <= pd.to_datetime(end)]
    return out


def _index_to_category(index: pd.Index) -> list[str]:
    if isinstance(index, pd.MultiIndex):
        labels = []
        for lvl0, lvl1 in index.tolist():
            d = (
                lvl0.strftime("%Y-%m-%d")
                if isinstance(lvl0, pd.Timestamp)
                else str(lvl0)
            )
            if isinstance(lvl1, pd.Timestamp):
                t = lvl1.strftime("%H:%M")
            elif hasattr(lvl1, "strftime"):
                t = lvl1.strftime("%H:%M")
            elif isinstance(lvl1, (int, np.integer, float, np.floating)):
                minutes = int(lvl1)
                if 0 <= minutes < 24 * 60:
                    t = f"{minutes // 60:02d}:{minutes % 60:02d}"
                else:
                    t = str(minutes)
            else:
                t = str(lvl1)
            labels.append(f"{d} {t}".strip())
        return labels
    if isinstance(index, pd.DatetimeIndex):
        return index.strftime("%Y-%m-%d %H:%M").tolist()
    return [str(x) for x in index.tolist()]


def _index_to_timestamps(index: pd.Index) -> list[pd.Timestamp]:
    if isinstance(index, pd.MultiIndex):
        out: list[pd.Timestamp] = []
        for date_val, minute_val in index.tolist():
            d = pd.to_datetime(date_val)
            minute_int = int(minute_val)
            out.append(d + pd.Timedelta(minutes=minute_int))
        return out
    if isinstance(index, pd.DatetimeIndex):
        return [pd.Timestamp(ts) for ts in index.to_pydatetime()]
    return [pd.Timestamp(i) for i in range(len(index))]


def _index_to_unix_seconds(index: pd.Index) -> list[int]:
    if isinstance(index, pd.MultiIndex):
        times: list[int] = []
        for date_val, minute_val in index.tolist():
            d = pd.to_datetime(date_val)
            minute_int = int(minute_val)
            ts = d + pd.Timedelta(minutes=minute_int)
            times.append(int(ts.timestamp()))
        return times
    if isinstance(index, pd.DatetimeIndex):
        return [int(ts.timestamp()) for ts in index.to_pydatetime()]
    # fallback: treat as categories (not ideal for interactive)
    return list(range(len(index)))


def _top_volume_profile_levels(
    ohlc: pd.DataFrame,
    *,
    n: int,
    bins: int,
    price_col: Optional[str],
    volume_col: str,
) -> Tuple[list[float], list[float]]:
    """Return top-N support/resistance levels (by volume) for the whole visible window.

    Supports are the N highest-volume bins with center <= last close; resistances are > last close.
    """
    if n <= 0 or ohlc.empty:
        return [], []

    if price_col is None:
        price = (ohlc["high"] + ohlc["low"] + ohlc["close"]) / 3.0
    else:
        price = ohlc[price_col]

    if volume_col in ohlc.columns:
        volume = ohlc[volume_col]
    elif "volume" in ohlc.columns:
        volume = ohlc["volume"]
    else:
        volume = pd.Series(1.0, index=ohlc.index)

    p = price.to_numpy(dtype=float, copy=False)
    v = volume.to_numpy(dtype=float, copy=False)
    mask = np.isfinite(p) & np.isfinite(v)
    if mask.sum() < 2:
        return [], []

    p = p[mask]
    v = v[mask]
    p_min = float(np.min(p))
    p_max = float(np.max(p))
    if not (np.isfinite(p_min) and np.isfinite(p_max)):
        return [], []
    if p_min == p_max:
        return [p_min], [p_max]

    bins_i = max(int(bins), 2)
    edges = np.linspace(p_min, p_max, bins_i + 1)
    idx = np.searchsorted(edges, p, side="right") - 1
    idx = np.clip(idx, 0, bins_i - 1)
    vol_bins = np.bincount(idx, weights=v, minlength=bins_i)
    centers = (edges[:-1] + edges[1:]) / 2.0

    last_close = float(ohlc["close"].to_numpy(dtype=float, copy=False)[-1])
    if not np.isfinite(last_close):
        last_close = float(np.nanmedian(p))

    below_mask = centers <= last_close
    above_mask = centers > last_close

    def pick(mask_arr: np.ndarray) -> list[float]:
        if not mask_arr.any():
            return []
        cand_idx = np.flatnonzero(mask_arr)
        # Sort by volume desc, then price asc (deterministic tie-break).
        items = [(float(vol_bins[i]), float(centers[i])) for i in cand_idx]
        items.sort(key=lambda x: (-x[0], x[1]))
        return [c[1] for c in items[: int(n)]]

    return pick(below_mask), pick(above_mask)


def _step_series(index: pd.Index, start_pos: int, value: float) -> pd.Series:
    arr = np.full(len(index), np.nan, dtype=float)
    if 0 <= start_pos < len(arr):
        arr[start_pos:] = float(value)
    return pd.Series(arr, index=index)


def _fib_ratio_tag(ratio: float) -> str:
    return f"{int(round(float(ratio) * 1000)):03d}"


def _minute_extreme_pos(
    ohlc: pd.DataFrame, date: pd.Timestamp, *, kind: str
) -> Optional[int]:
    if not isinstance(ohlc.index, pd.MultiIndex):
        return None
    dates = pd.to_datetime(ohlc.index.get_level_values(0))
    mask = dates == pd.to_datetime(date)
    pos = np.flatnonzero(np.asarray(mask))
    if pos.size == 0:
        return None
    if kind == "low":
        vals = ohlc["low"].to_numpy(dtype=float, copy=False)[pos]
        if np.isfinite(vals).any():
            return int(pos[int(np.nanargmin(vals))])
        return int(pos[0])
    if kind == "high":
        vals = ohlc["high"].to_numpy(dtype=float, copy=False)[pos]
        if np.isfinite(vals).any():
            return int(pos[int(np.nanargmax(vals))])
        return int(pos[0])
    raise ValueError(f"Unknown kind: {kind!r}")


def _daily_pos(ohlc: pd.DataFrame, date: pd.Timestamp) -> Optional[int]:
    if isinstance(ohlc.index, pd.MultiIndex):
        return None
    idx = pd.to_datetime(ohlc.index)
    loc = idx.get_indexer([pd.to_datetime(date)])
    if loc.size == 0:
        return None
    pos = int(loc[0])
    return None if pos < 0 else pos


def _fib_marker_info(
    ohlc: pd.DataFrame, fib: dict[str, object]
) -> Optional[dict[str, object]]:
    try:
        low_date = pd.to_datetime(fib["low_date"])
        high_date = pd.to_datetime(fib["high_date"])
        fib_low = float(fib["low"])
        fib_high = float(fib["high"])
    except Exception:
        return None

    if isinstance(ohlc.index, pd.MultiIndex):
        low_pos = _minute_extreme_pos(ohlc, low_date, kind="low")
        high_pos = _minute_extreme_pos(ohlc, high_date, kind="high")
    else:
        low_pos = _daily_pos(ohlc, low_date)
        high_pos = _daily_pos(ohlc, high_date)

    if low_pos is None or high_pos is None:
        return None

    low_price = float(ohlc["low"].iloc[int(low_pos)])
    high_price = float(ohlc["high"].iloc[int(high_pos)])
    return {
        "low_pos": int(low_pos),
        "high_pos": int(high_pos),
        "low_date": low_date,
        "high_date": high_date,
        "fib_low": float(fib_low),
        "fib_high": float(fib_high),
        "fib_delta": float(fib_high - fib_low),
        "low_price": low_price,
        "high_price": high_price,
        "view_delta": float(high_price - low_price),
    }


def _compute_fib_overlay(
    ohlc: pd.DataFrame,
    *,
    daily_ohlc: Optional[pd.DataFrame],
    lookaround_days: int,
    min_high_mult: float,
    min_high_points: float,
    ratios: Iterable[float],
    first_ratio: float,
) -> Tuple[dict[float, pd.Series], Optional[dict[str, object]]]:
    source_daily = daily_ohlc if daily_ohlc is not None else ohlc
    if isinstance(source_daily.index, pd.MultiIndex):
        return {}, None

    # Ensure the swing (low/high/touch) is derived from data that is actually visible.
    # Otherwise it may "remember" an old all-time-high outside the current view.
    if isinstance(ohlc.index, pd.MultiIndex):
        view_dates = pd.to_datetime(ohlc.index.get_level_values(0))
        view_start = pd.to_datetime(view_dates.min())
        view_end = pd.to_datetime(view_dates.max())
        if daily_ohlc is not None:
            sliced = daily_ohlc.loc[(daily_ohlc.index >= view_start) & (daily_ohlc.index <= view_end)]
            if not sliced.empty:
                source_daily = sliced
    fib = fibonacci_swing_support(
        source_daily,
        lookaround_days=int(lookaround_days),
        min_high_mult=float(min_high_mult),
        min_high_points=float(min_high_points),
        ratios=tuple(float(r) for r in ratios),
        first_ratio=float(first_ratio),
    )
    if fib is None:
        return {}, None

    levels = fib.get("levels")
    if not isinstance(levels, dict) or not levels:
        return {}, fib

    first_level = levels.get(float(first_ratio))
    if first_level is None or not np.isfinite(float(first_level)):
        return {}, fib

    start_pos: Optional[int] = None
    if isinstance(ohlc.index, pd.MultiIndex):
        dates = pd.to_datetime(ohlc.index.get_level_values(0))
        view_start_date = pd.to_datetime(dates.min())
        touch_date = pd.to_datetime(fib["touch_date"])
        if touch_date <= view_start_date:
            start_pos = 0
        else:
            high_date = pd.to_datetime(fib["high_date"])
            low_vals = ohlc["low"].to_numpy(dtype=float, copy=False)
            high_vals = ohlc["high"].to_numpy(dtype=float, copy=False)

            if high_date < view_start_date:
                search_start = 0
            else:
                high_day_mask = dates == high_date
                high_day_pos = np.flatnonzero(np.asarray(high_day_mask))
                if high_day_pos.size:
                    day_high = high_vals[high_day_pos]
                    if np.isfinite(day_high).any():
                        max_in_day = int(high_day_pos[int(np.nanargmax(day_high))])
                    else:
                        max_in_day = int(high_day_pos[0])
                    search_start = max_in_day
                else:
                    search_start = int(
                        np.searchsorted(dates.to_numpy(), np.datetime64(high_date), side="left")
                    )
                    search_start = int(np.clip(search_start, 0, len(dates)))

            slice_low = low_vals[search_start:]
            cond = np.isfinite(slice_low) & (slice_low <= float(first_level))
            if cond.any():
                start_pos = int(search_start + int(np.argmax(cond)))
            else:
                start_pos = None
    else:
        touch_date = pd.to_datetime(fib["touch_date"])
        idx = pd.to_datetime(ohlc.index)
        start_pos = int(idx.searchsorted(touch_date, side="left"))
        start_pos = int(np.clip(start_pos, 0, len(idx)))

    if start_pos is None or start_pos >= len(ohlc):
        return {}, fib

    out: dict[float, pd.Series] = {}
    for ratio, level in levels.items():
        try:
            ratio_f = float(ratio)
            level_f = float(level)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(level_f):
            continue
        out[ratio_f] = _step_series(ohlc.index, start_pos, level_f)
    return out, fib


def plot_kline(
    ohlc: pd.DataFrame,
    output_dir: Path,
    filename: str,
    *,
    show_top_sr: bool = False,
    top_sr_n: int = 3,
    show_topk_poc: bool = False,
    poc_topk: int = 3,
    show_poc: bool = False,
    poc_window: int = 240,
    poc_windows: Optional[Iterable[int]] = None,
    poc_bins: int = 40,
    poc_shift: int = 1,
    poc_groupby_level: Optional[int] = None,
    show_daily_poc: bool = False,
    daily_poc_window: int = 60,
    daily_poc_windows: Optional[Iterable[int]] = None,
    daily_poc_bins: int = 40,
    daily_poc_shift: int = 1,
    poc_price_col: Optional[str] = None,
    poc_volume_col: str = "volume",
    show_fib: bool = False,
    fib_lookaround_days: int = 3,
    fib_min_high_mult: float = 1.3,
    fib_min_high_points: float = 0.0,
    fib_ratios: Optional[Iterable[float]] = None,
    fib_first_ratio: float = 0.236,
    daily_ohlc: Optional[pd.DataFrame] = None,
) -> Path:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required to plot K-line. "
            "Install it (e.g. `pip install matplotlib`) and rerun."
        ) from e

    if ohlc.empty:
        raise ValueError("No OHLC data to plot.")

    output_dir.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(ohlc))
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    width = 0.6

    for i, row in enumerate(ohlc.itertuples()):
        open_ = row.open
        close = row.close
        high = row.high
        low = row.low
        color = "#2f6fb0" if close >= open_ else "#d9534f"

        ax.plot([i, i], [low, high], color=color, linewidth=1.0)
        body_low = min(open_, close)
        body_high = max(open_, close)
        rect = Rectangle((i - width / 2.0, body_low), width, body_high - body_low, facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    if show_fib:
        fib_series, fib_info = _compute_fib_overlay(
            ohlc,
            daily_ohlc=daily_ohlc,
            lookaround_days=fib_lookaround_days,
            min_high_mult=fib_min_high_mult,
            min_high_points=fib_min_high_points,
            ratios=tuple(fib_ratios) if fib_ratios is not None else (0.236, 0.382, 0.5, 0.618),
            first_ratio=fib_first_ratio,
        )
        if fib_series:
            palette = [
                "#6f42c1",
                "#fd7e14",
                "#20c997",
                "#17a2b8",
                "#ffc107",
            ]
            for j, (ratio, series) in enumerate(sorted(fib_series.items(), key=lambda kv: kv[0])):
                color = palette[j % len(palette)]
                ax.plot(
                    x,
                    series.to_numpy(dtype=float, copy=False),
                    color=color,
                    linewidth=1.2,
                    linestyle="--",
                    alpha=0.85,
                    label=f"fib_{_fib_ratio_tag(ratio)}",
                )
            ax.legend(loc="best", fontsize=8)

        if fib_info is not None:
            marker = _fib_marker_info(ohlc, fib_info)
            if marker is not None:
                lp = int(marker["low_pos"])
                hp = int(marker["high_pos"])
                low_price = float(marker["low_price"])
                high_price = float(marker["high_price"])
                view_delta = float(marker["view_delta"])
                fib_delta = float(marker["fib_delta"])
                ax.scatter([lp], [low_price], color="#198754", s=40, zorder=5)
                ax.scatter([hp], [high_price], color="#dc3545", s=40, zorder=5)
                ax.annotate(
                    f"LOW {low_price:.2f}",
                    (lp, low_price),
                    xytext=(0, -18),
                    textcoords="offset points",
                    ha="center",
                    va="top",
                    fontsize=9,
                    color="#198754",
                )
                ax.annotate(
                    f"HIGH {high_price:.2f}\nΔ(view) {view_delta:.2f}  Δ(fib) {fib_delta:.2f}",
                    (hp, high_price),
                    xytext=(0, 12),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="#dc3545",
                )

    if show_top_sr:
        profile_src = daily_ohlc if isinstance(ohlc.index, pd.MultiIndex) and daily_ohlc is not None else ohlc
        supports, resistances = _top_volume_profile_levels(
            profile_src,
            n=int(top_sr_n),
            bins=int(poc_bins),
            price_col=poc_price_col,
            volume_col=poc_volume_col,
        )
        for j, lvl in enumerate(supports, start=1):
            ax.axhline(
                y=float(lvl),
                color="#ff9f40",
                linewidth=1.2,
                linestyle="-",
                alpha=0.65,
                label=f"top_support_{j}",
                zorder=2,
            )
        # NOTE: resistances (above) intentionally not drawn to keep the chart readable.

    if show_topk_poc and int(poc_topk) > 1:
        windows = tuple(int(w) for w in (poc_windows or (poc_window,)))
        for window in windows:
            below_lvls, above_lvls = volume_profile_poc_topk_levels(
                ohlc,
                window=window,
                bins=int(poc_bins),
                k=int(poc_topk),
                price_col=poc_price_col,
                volume_col=poc_volume_col,
                min_periods=1,
                shift=int(poc_shift),
                groupby_level=poc_groupby_level,
            )
            # Plot top-k supports/resistances; k=1 is roughly the existing poc_below/poc_above.
            for j, s in enumerate(below_lvls, start=1):
                ax.plot(
                    x,
                    s.to_numpy(dtype=float, copy=False),
                    color="#ff9f40",
                    linewidth=1.0,
                    linestyle="--",
                    alpha=0.55,
                    label=f"poc_below_{window}_top{j}",
                    zorder=3,
                )
            # Nearest support/resistance among the top-k-by-volume candidates.
            if below_lvls:
                nearest_below = pd.concat(below_lvls, axis=1).max(axis=1, skipna=True)
                ax.plot(
                    x,
                    nearest_below.to_numpy(dtype=float, copy=False),
                    color="#ff9f40",
                    linewidth=1.4,
                    linestyle="-",
                    alpha=0.75,
                    label=f"poc_below_{window}_nearest",
                    zorder=3,
                )

    # Overlay POC after Fib (on top), as requested.
    if show_poc:
        windows = tuple(int(w) for w in (poc_windows or (poc_window,)))
        palette = [
            ("#6f42c1", "--"),
            ("#dc3545", "--"),
            ("#17a2b8", "--"),
            ("#28a745", "--"),
        ]
        for i, window in enumerate(windows):
            color, style = palette[i % len(palette)]
            poc_below, poc_above = volume_profile_poc_levels(
                ohlc,
                window=window,
                bins=int(poc_bins),
                price_col=poc_price_col,
                volume_col=poc_volume_col,
                min_periods=1,
                shift=int(poc_shift),
                groupby_level=poc_groupby_level,
            )
            ax.plot(
                x,
                poc_below.to_numpy(dtype=float, copy=False),
                color=color,
                linewidth=1.4,
                linestyle=style,
                alpha=0.95,
                label=f"poc_below_{window}",
                zorder=4,
            )
            ax.plot(
                x,
                poc_above.to_numpy(dtype=float, copy=False),
                color=color,
                linewidth=1.4,
                linestyle=":",
                alpha=0.95,
                label=f"poc_above_{window}",
                zorder=4,
            )

    if show_daily_poc and daily_ohlc is not None:
        windows = tuple(int(w) for w in (daily_poc_windows or (daily_poc_window,)))
        palette = [
            ("#ff9f40", "-"),
            ("#ffc107", "-"),
            ("#fd7e14", "--"),
            ("#e83e8c", "--"),
        ]
        for i, window in enumerate(windows):
            color, style = palette[i % len(palette)]
            d_poc_below, d_poc_above = volume_profile_poc_levels(
                daily_ohlc,
                window=window,
                bins=int(daily_poc_bins),
                price_col=poc_price_col,
                volume_col=poc_volume_col,
                min_periods=1,
                shift=int(daily_poc_shift),
                groupby_level=None,
            )
            # Show top-k daily supports/resistances as time-varying curves (especially useful for 120d).
            if int(poc_topk) > 1:
                d_b_lvls, d_a_lvls = volume_profile_poc_topk_levels(
                    daily_ohlc,
                    window=window,
                    bins=int(daily_poc_bins),
                    k=int(poc_topk),
                    price_col=poc_price_col,
                    volume_col=poc_volume_col,
                    min_periods=1,
                    shift=int(daily_poc_shift),
                    groupby_level=None,
                )
                if isinstance(ohlc.index, pd.MultiIndex):
                    d_b_lvls = [_broadcast_daily_to_minutes(s, ohlc.index) for s in d_b_lvls]
                    # Resistances (above) intentionally not drawn to keep the chart readable.
                for j, s in enumerate(d_b_lvls, start=1):
                    ax.plot(
                        x,
                        s.to_numpy(dtype=float, copy=False),
                        color="#ff9f40",
                        linewidth=1.0,
                        linestyle="--",
                        alpha=0.45,
                        label=f"daily_poc_below_{window}_top{j}",
                        zorder=3,
                    )
                if d_b_lvls:
                    nearest_below = pd.concat(d_b_lvls, axis=1).max(axis=1, skipna=True)
                    ax.plot(
                        x,
                        nearest_below.to_numpy(dtype=float, copy=False),
                        color="#ff9f40",
                        linewidth=1.6,
                        linestyle="-",
                        alpha=0.8,
                        label=f"daily_poc_below_{window}_nearest",
                        zorder=3,
                    )
            if isinstance(ohlc.index, pd.MultiIndex):
                d_poc_below = _broadcast_daily_to_minutes(d_poc_below, ohlc.index)
                d_poc_above = _broadcast_daily_to_minutes(d_poc_above, ohlc.index)
            base_alpha = 0.35 if int(poc_topk) > 1 else 0.9
            base_lw = 1.0 if int(poc_topk) > 1 else 1.8
            ax.plot(
                x,
                d_poc_below.to_numpy(dtype=float, copy=False),
                color=color,
                linewidth=base_lw,
                linestyle=style,
                alpha=base_alpha,
                label=f"daily_poc_below_{window}",
                zorder=4,
            )
            ax.plot(
                x,
                d_poc_above.to_numpy(dtype=float, copy=False),
                color=color,
                linewidth=base_lw,
                linestyle=":",
                alpha=base_alpha,
                label=f"daily_poc_above_{window}",
                zorder=4,
            )

    if (show_poc or show_daily_poc) and ax.get_legend() is None:
        ax.legend(loc="best", fontsize=8)

    step = max(len(ohlc) // 10, 1)
    ticks = list(range(0, len(ohlc), step))
    labels = []
    for i in ticks:
        idx = ohlc.index[i]
        if isinstance(idx, pd.Timestamp):
            labels.append(idx.strftime("%Y-%m-%d"))
        elif isinstance(idx, tuple) and idx:
            labels.append(str(idx[0]))
        else:
            labels.append(str(idx))
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title("K-line (Minute)" if isinstance(ohlc.index, pd.MultiIndex) else "K-line (Daily)")
    ax.set_ylabel("Price")
    ax.grid(alpha=0.2)
    fig.tight_layout()

    out_path = output_dir / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_kline_interactive(
    ohlc: pd.DataFrame,
    output_dir: Path,
    filename: str,
    renderer: str = "plotly",
    show_volume_zone: bool = False,
    volume_zone_window: int = 20,
    volume_zone_mult: float = 1.0,
    show_top_sr: bool = False,
    top_sr_n: int = 3,
    show_topk_poc: bool = False,
    poc_topk: int = 3,
    show_poc: bool = True,
    poc_window: int = 240,
    poc_windows: Optional[Iterable[int]] = None,
    poc_bins: int = 40,
    poc_shift: int = 1,
    poc_groupby_level: Optional[int] = None,
    show_daily_poc: bool = True,
    daily_poc_window: int = 60,
    daily_poc_windows: Optional[Iterable[int]] = None,
    daily_poc_bins: int = 40,
    daily_poc_shift: int = 1,
    poc_price_col: Optional[str] = None,
    poc_volume_col: str = "volume",
    show_fib: bool = False,
    fib_lookaround_days: int = 3,
    fib_min_high_mult: float = 1.3,
    fib_min_high_points: float = 0.0,
    fib_ratios: Optional[Iterable[float]] = None,
    fib_first_ratio: float = 0.236,
    fib_daily_ohlc: Optional[pd.DataFrame] = None,
    daily_ohlc: Optional[pd.DataFrame] = None,
) -> Path:
    renderer = str(renderer).lower().strip()
    if renderer in ("lw", "lightweight", "lightweight-charts", "lightweight_charts"):
        return _plot_kline_interactive_lightweight(
            ohlc,
            output_dir,
            filename,
            show_poc=show_poc,
            poc_window=poc_window,
            poc_windows=poc_windows,
            poc_bins=poc_bins,
            poc_shift=poc_shift,
            poc_groupby_level=poc_groupby_level,
            show_daily_poc=show_daily_poc,
            daily_poc_window=daily_poc_window,
            daily_poc_windows=daily_poc_windows,
            daily_poc_bins=daily_poc_bins,
            daily_poc_shift=daily_poc_shift,
            poc_price_col=poc_price_col,
            poc_volume_col=poc_volume_col,
            show_fib=show_fib,
            fib_lookaround_days=fib_lookaround_days,
            fib_min_high_mult=fib_min_high_mult,
            fib_min_high_points=fib_min_high_points,
            fib_ratios=fib_ratios,
            fib_first_ratio=fib_first_ratio,
            fib_daily_ohlc=fib_daily_ohlc,
            daily_ohlc=daily_ohlc,
        )

    if renderer not in ("plotly",):
        raise ValueError(f"Unknown interactive renderer: {renderer!r}. Use 'plotly' or 'lw'.")

    try:
        import plotly.graph_objects as go
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "plotly is required for interactive K-line. "
            "Install it (e.g. `pip install plotly`) and rerun."
        ) from e

    if ohlc.empty:
        raise ValueError("No OHLC data to plot.")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Use categories to avoid inserting gaps for non-trading hours/days.
    x_vals = _index_to_category(ohlc.index)
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=x_vals,
                open=ohlc["open"],
                high=ohlc["high"],
                low=ohlc["low"],
                close=ohlc["close"],
                increasing_line_color="#2f6fb0",
                decreasing_line_color="#d9534f",
                name="price",
            )
        ]
    )

    # Draw Fib first, then overlay POC lines on top (as requested).
    if show_fib:
        fib_series, _ = _compute_fib_overlay(
            ohlc,
            daily_ohlc=fib_daily_ohlc if fib_daily_ohlc is not None else daily_ohlc,
            lookaround_days=fib_lookaround_days,
            min_high_mult=fib_min_high_mult,
            min_high_points=fib_min_high_points,
            ratios=tuple(fib_ratios) if fib_ratios is not None else (0.236, 0.382, 0.5, 0.618),
            first_ratio=fib_first_ratio,
        )
        if fib_series:
            palette = [
                ("rgba(111, 66, 193, 0.9)", "dash"),
                ("rgba(253, 126, 20, 0.9)", "dash"),
                ("rgba(32, 201, 151, 0.9)", "dash"),
                ("rgba(23, 162, 184, 0.9)", "dash"),
                ("rgba(255, 193, 7, 0.9)", "dash"),
            ]
            for j, (ratio, series) in enumerate(sorted(fib_series.items(), key=lambda kv: kv[0])):
                color, dash = palette[j % len(palette)]
                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=series,
                        mode="lines",
                        line=dict(color=color, width=1.6, dash=dash),
                        name=f"fib_{_fib_ratio_tag(ratio)}",
                    )
                )

    if show_top_sr:
        profile_src = daily_ohlc if isinstance(ohlc.index, pd.MultiIndex) and daily_ohlc is not None else ohlc
        supports, resistances = _top_volume_profile_levels(
            profile_src,
            n=int(top_sr_n),
            bins=int(poc_bins),
            price_col=poc_price_col,
            volume_col=poc_volume_col,
        )
        for j, lvl in enumerate(supports, start=1):
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=[float(lvl)] * len(x_vals),
                    mode="lines",
                    line=dict(color="rgba(255, 159, 64, 0.6)", width=1.4),
                    name=f"top_support_{j}",
                )
            )
        # NOTE: resistances (above) intentionally not drawn to keep the chart readable.

    if show_topk_poc and int(poc_topk) > 1:
        windows = tuple(int(w) for w in (poc_windows or (poc_window,)))
        for window in windows:
            below_lvls, above_lvls = volume_profile_poc_topk_levels(
                ohlc,
                window=window,
                bins=int(poc_bins),
                k=int(poc_topk),
                price_col=poc_price_col,
                volume_col=poc_volume_col,
                min_periods=1,
                shift=int(poc_shift),
                groupby_level=poc_groupby_level,
            )
            for j, s in enumerate(below_lvls, start=1):
                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=s,
                        mode="lines",
                        line=dict(color="rgba(255, 159, 64, 0.45)", width=1.0, dash="dot"),
                        name=f"poc_below_{window}_top{j}",
                    )
                )
            if below_lvls:
                nearest_below = pd.concat(below_lvls, axis=1).max(axis=1, skipna=True)
                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=nearest_below,
                        mode="lines",
                        line=dict(color="rgba(255, 159, 64, 0.8)", width=1.6),
                        name=f"poc_below_{window}_nearest",
                    )
                )

    if show_poc:
        windows = tuple(int(w) for w in (poc_windows or (poc_window,)))
        palette = [
            ("rgba(111, 66, 193, 0.9)", "dot"),
            ("rgba(220, 53, 69, 0.9)", "dot"),
            ("rgba(23, 162, 184, 0.9)", "dot"),
            ("rgba(40, 167, 69, 0.9)", "dot"),
        ]
        for i, window in enumerate(windows):
            color, dash = palette[i % len(palette)]
            poc_below, poc_above = volume_profile_poc_levels(
                ohlc,
                window=window,
                bins=int(poc_bins),
                price_col=poc_price_col,
                volume_col=poc_volume_col,
                min_periods=1,
                shift=int(poc_shift),
                groupby_level=poc_groupby_level,
            )
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=poc_below,
                    mode="lines",
                    line=dict(color=color, width=1.2, dash=dash),
                    name=f"poc_below_{window}",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=poc_above,
                    mode="lines",
                    line=dict(color=color, width=1.2, dash="dash"),
                    name=f"poc_above_{window}",
                )
            )

    if show_daily_poc and daily_ohlc is not None:
        windows = tuple(int(w) for w in (daily_poc_windows or (daily_poc_window,)))
        palette = [
            ("rgba(255, 159, 64, 0.9)", "solid"),
            ("rgba(255, 193, 7, 0.9)", "solid"),
            ("rgba(255, 159, 64, 0.9)", "dash"),
            ("rgba(255, 193, 7, 0.9)", "dash"),
        ]
        for i, window in enumerate(windows):
            color, dash = palette[i % len(palette)]
            d_poc_below, d_poc_above = volume_profile_poc_levels(
                daily_ohlc,
                window=window,
                bins=int(daily_poc_bins),
                price_col=poc_price_col,
                volume_col=poc_volume_col,
                min_periods=1,
                shift=int(daily_poc_shift),
                groupby_level=None,
            )
            if int(poc_topk) > 1:
                d_b_lvls, d_a_lvls = volume_profile_poc_topk_levels(
                    daily_ohlc,
                    window=window,
                    bins=int(daily_poc_bins),
                    k=int(poc_topk),
                    price_col=poc_price_col,
                    volume_col=poc_volume_col,
                    min_periods=1,
                    shift=int(daily_poc_shift),
                    groupby_level=None,
                )
                if isinstance(ohlc.index, pd.MultiIndex):
                    d_b_lvls = [_broadcast_daily_to_minutes(s, ohlc.index) for s in d_b_lvls]
                    # Resistances (above) intentionally not drawn to keep the chart readable.
                for j, s in enumerate(d_b_lvls, start=1):
                    fig.add_trace(
                        go.Scatter(
                            x=x_vals,
                            y=s,
                            mode="lines",
                            line=dict(color="rgba(255, 159, 64, 0.35)", width=1.0, dash="dot"),
                            name=f"daily_poc_below_{window}_top{j}",
                        )
                    )
                if d_b_lvls:
                    nearest_below = pd.concat(d_b_lvls, axis=1).max(axis=1, skipna=True)
                    fig.add_trace(
                        go.Scatter(
                            x=x_vals,
                            y=nearest_below,
                            mode="lines",
                            line=dict(color="rgba(255, 159, 64, 0.85)", width=2.0),
                            name=f"daily_poc_below_{window}_nearest",
                        )
                    )
            if isinstance(ohlc.index, pd.MultiIndex):
                d_poc_below = _broadcast_daily_to_minutes(d_poc_below, ohlc.index)
                d_poc_above = _broadcast_daily_to_minutes(d_poc_above, ohlc.index)
            base_color = "rgba(255, 159, 64, 0.25)" if int(poc_topk) > 1 else color
            base_width = 1.0 if int(poc_topk) > 1 else 1.4
            base_dash = "dot" if int(poc_topk) > 1 else dash
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=d_poc_below,
                    mode="lines",
                    line=dict(color=base_color, width=base_width, dash=base_dash),
                    name=f"daily_poc_below_{window}",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=d_poc_above,
                    mode="lines",
                    line=dict(color=base_color, width=base_width, dash="dot" if int(poc_topk) > 1 else "dash"),
                    name=f"daily_poc_above_{window}",
                )
            )

    fig.update_layout(
        title="K-line (Minute)" if isinstance(ohlc.index, pd.MultiIndex) else "K-line (Daily)",
        xaxis_title="date",
        yaxis_title="price",
        template="plotly_white",
        dragmode="pan",
        xaxis_rangeslider_visible=True,
        xaxis_type="category",
        height=600,
    )
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.07))

    out_path = output_dir / filename
    # Embed plotly.js so the HTML works offline.
    fig.write_html(out_path, include_plotlyjs="include")
    return out_path


def _series_to_lw_line_data(times: list[int], values: pd.Series) -> list[dict[str, object]]:
    arr = values.to_numpy(dtype=float, copy=False)
    out: list[dict[str, object]] = []
    for t, v in zip(times, arr):
        if np.isfinite(v):
            out.append({"time": int(t), "value": float(v)})
    return out


def _lw_markers_from_fib(
    times: list[int], ohlc: pd.DataFrame, fib: Optional[dict[str, object]]
) -> list[dict[str, object]]:
    if fib is None:
        return []
    marker = _fib_marker_info(ohlc, fib)
    if marker is None:
        return []
    lp = int(marker["low_pos"])
    hp = int(marker["high_pos"])
    low_price = float(marker["low_price"])
    high_price = float(marker["high_price"])
    view_delta = float(marker["view_delta"])
    fib_delta = float(marker["fib_delta"])
    if not (0 <= lp < len(times) and 0 <= hp < len(times)):
        return []
    return [
        {
            "time": int(times[lp]),
            "position": "belowBar",
            "color": "#198754",
            "shape": "arrowUp",
            "text": f"LOW {low_price:.2f}",
        },
        {
            "time": int(times[hp]),
            "position": "aboveBar",
            "color": "#dc3545",
            "shape": "arrowDown",
            "text": f"HIGH {high_price:.2f}  Δ(view){view_delta:.2f}  Δ(fib){fib_delta:.2f}",
        },
    ]


def _plot_kline_interactive_lightweight(
    ohlc: pd.DataFrame,
    output_dir: Path,
    filename: str,
    *,
    show_poc: bool,
    poc_window: int,
    poc_windows: Optional[Iterable[int]],
    poc_bins: int,
    poc_shift: int,
    poc_groupby_level: Optional[int],
    show_daily_poc: bool,
    daily_poc_window: int,
    daily_poc_windows: Optional[Iterable[int]],
    daily_poc_bins: int,
    daily_poc_shift: int,
    poc_price_col: Optional[str],
    poc_volume_col: str,
    show_fib: bool,
    fib_lookaround_days: int,
    fib_min_high_mult: float,
    fib_min_high_points: float,
    fib_ratios: Optional[Iterable[float]],
    fib_first_ratio: float,
    fib_daily_ohlc: Optional[pd.DataFrame],
    daily_ohlc: Optional[pd.DataFrame],
) -> Path:
    if ohlc.empty:
        raise ValueError("No OHLC data to plot.")
    output_dir.mkdir(parents=True, exist_ok=True)

    times = _index_to_unix_seconds(ohlc.index)
    candles = []
    for t, row in zip(times, ohlc.itertuples()):
        candles.append(
            {
                "time": int(t),
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
            }
        )

    line_series_payload: list[dict[str, object]] = []
    legend_items: list[dict[str, str]] = []
    markers: list[dict[str, object]] = []

    # Build Fib first, then overlay POC lines on top (as requested).
    if show_fib:
        fib_source = fib_daily_ohlc if fib_daily_ohlc is not None else daily_ohlc
        fib_series, fib_info = _compute_fib_overlay(
            ohlc,
            daily_ohlc=fib_source,
            lookaround_days=fib_lookaround_days,
            min_high_mult=fib_min_high_mult,
            min_high_points=fib_min_high_points,
            ratios=tuple(fib_ratios) if fib_ratios is not None else (0.236, 0.382, 0.5, 0.618),
            first_ratio=fib_first_ratio,
        )
        markers = _lw_markers_from_fib(times, ohlc, fib_info)
        palette = ["#6f42c1", "#fd7e14", "#20c997", "#17a2b8", "#ffc107"]
        for j, (ratio, series) in enumerate(sorted(fib_series.items(), key=lambda kv: kv[0])):
            color = palette[j % len(palette)]
            data = _series_to_lw_line_data(times, series)
            if not data:
                continue
            name = f"fib_{_fib_ratio_tag(ratio)}"
            line_series_payload.append(
                {
                    "name": name,
                    "color": color,
                    "data": data,
                    "style": {"lineWidth": 2, "lineStyle": 2},
                }
            )
            legend_items.append({"name": name, "color": color})

    if show_poc:
        windows = tuple(int(w) for w in (poc_windows or (poc_window,)))
        palette = [
            ("#6f42c1", "poc_below"),
            ("#dc3545", "poc_above"),
            ("#17a2b8", "poc_below"),
            ("#28a745", "poc_above"),
        ]
        for i, window in enumerate(windows):
            color_below, _ = palette[(2 * i) % len(palette)]
            color_above, _ = palette[(2 * i + 1) % len(palette)]
            poc_below, poc_above = volume_profile_poc_levels(
                ohlc,
                window=window,
                bins=int(poc_bins),
                price_col=poc_price_col,
                volume_col=poc_volume_col,
                shift=int(poc_shift),
                groupby_level=poc_groupby_level,
            )
            below_data = _series_to_lw_line_data(times, poc_below)
            above_data = _series_to_lw_line_data(times, poc_above)
            name_below = f"poc_below_{window}"
            name_above = f"poc_above_{window}"
            if below_data:
                line_series_payload.append(
                    {
                        "name": name_below,
                        "color": color_below,
                        "data": below_data,
                        "style": {"lineWidth": 1, "lineStyle": 2},
                    }
                )
                legend_items.append({"name": name_below, "color": color_below})
            if above_data:
                line_series_payload.append(
                    {
                        "name": name_above,
                        "color": color_above,
                        "data": above_data,
                        "style": {"lineWidth": 1, "lineStyle": 2},
                    }
                )
                legend_items.append({"name": name_above, "color": color_above})

    if show_daily_poc and daily_ohlc is not None:
        windows = tuple(int(w) for w in (daily_poc_windows or (daily_poc_window,)))
        palette = [
            "#ff9f40",
            "#ffc107",
            "#fd7e14",
            "#e83e8c",
        ]
        for i, window in enumerate(windows):
            color = palette[i % len(palette)]
            d_poc_below, d_poc_above = volume_profile_poc_levels(
                daily_ohlc,
                window=window,
                bins=int(daily_poc_bins),
                price_col=poc_price_col,
                volume_col=poc_volume_col,
                shift=int(daily_poc_shift),
                groupby_level=None,
            )
            if isinstance(ohlc.index, pd.MultiIndex):
                d_poc_below = _broadcast_daily_to_minutes(d_poc_below, ohlc.index)
                d_poc_above = _broadcast_daily_to_minutes(d_poc_above, ohlc.index)
            below_data = _series_to_lw_line_data(times, d_poc_below)
            above_data = _series_to_lw_line_data(times, d_poc_above)
            name_below = f"daily_poc_below_{window}"
            name_above = f"daily_poc_above_{window}"
            if below_data:
                line_series_payload.append(
                    {
                        "name": name_below,
                        "color": color,
                        "data": below_data,
                        "style": {"lineWidth": 2, "lineStyle": 0},
                    }
                )
                legend_items.append({"name": name_below, "color": color})
            if above_data:
                line_series_payload.append(
                    {
                        "name": name_above,
                        "color": color,
                        "data": above_data,
                        "style": {"lineWidth": 2, "lineStyle": 3},
                    }
                )
                legend_items.append({"name": name_above, "color": color})

    title = "K-line (Minute)" if isinstance(ohlc.index, pd.MultiIndex) else "K-line (Daily)"
    payload = {
        "title": title,
        "candles": candles,
        "lines": line_series_payload,
        "legend": legend_items,
        "markers": markers,
    }

    # Lightweight Charts via CDN; user said opening method doesn't matter.
    # If you later want offline, we can vendor the JS file and reference it locally.
    html = f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{title}</title>
    <style>
      html, body {{ margin: 0; padding: 0; height: 100%; background: #ffffff; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial; }}
      #wrap {{ position: relative; height: 100%; }}
      #chart {{ position: absolute; inset: 0; }}
      #legend {{
        position: absolute; top: 10px; left: 10px;
        background: rgba(255,255,255,0.9);
        border: 1px solid rgba(0,0,0,0.08);
        border-radius: 8px;
        padding: 8px 10px;
        max-width: 380px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.08);
        font-size: 12px;
        line-height: 1.25;
      }}
      #legend .row {{ display: inline-flex; align-items: center; margin: 2px 8px 2px 0; white-space: nowrap; }}
      #legend .swatch {{ width: 10px; height: 2px; margin-right: 6px; border-radius: 1px; }}
      #legend .title {{ font-weight: 600; margin-bottom: 6px; }}
    </style>
  </head>
  <body>
    <div id="wrap">
      <div id="chart"></div>
      <div id="legend"></div>
    </div>
    <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
    <script>
      const payload = {json.dumps(payload, ensure_ascii=False)};
      const el = document.getElementById('chart');
      const legend = document.getElementById('legend');

      const chart = LightweightCharts.createChart(el, {{
        layout: {{ background: {{ color: '#ffffff' }}, textColor: '#1f2937' }},
        grid: {{ vertLines: {{ color: 'rgba(31,41,55,0.06)' }}, horzLines: {{ color: 'rgba(31,41,55,0.06)' }} }},
        crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
        rightPriceScale: {{ borderColor: 'rgba(31,41,55,0.15)' }},
        timeScale: {{ borderColor: 'rgba(31,41,55,0.15)', timeVisible: true, secondsVisible: false }},
      }});

      const candleSeries = chart.addCandlestickSeries({{
        upColor: '#2f6fb0', downColor: '#d9534f',
        borderUpColor: '#2f6fb0', borderDownColor: '#d9534f',
        wickUpColor: '#2f6fb0', wickDownColor: '#d9534f',
      }});
      candleSeries.setData(payload.candles);
      if (payload.markers && payload.markers.length) {{
        candleSeries.setMarkers(payload.markers);
      }}

      for (const line of payload.lines) {{
        const s = chart.addLineSeries({{
          color: line.color,
          lineWidth: (line.style && line.style.lineWidth) ? line.style.lineWidth : 2,
          lineStyle: (line.style && line.style.lineStyle) ? line.style.lineStyle : 0,
          priceLineVisible: false,
          lastValueVisible: false,
        }});
        s.setData(line.data);
      }}

      legend.innerHTML = '';
      const title = document.createElement('div');
      title.className = 'title';
      title.textContent = payload.title;
      legend.appendChild(title);
      for (const item of payload.legend) {{
        const row = document.createElement('div');
        row.className = 'row';
        const sw = document.createElement('span');
        sw.className = 'swatch';
        sw.style.background = item.color;
        const name = document.createElement('span');
        name.textContent = item.name;
        row.appendChild(sw);
        row.appendChild(name);
        legend.appendChild(row);
      }}

      chart.timeScale().fitContent();
      window.addEventListener('resize', () => {{
        chart.applyOptions({{ width: el.clientWidth, height: el.clientHeight }});
      }});
    </script>
  </body>
</html>
"""

    out_path = output_dir / filename
    out_path.write_text(html, encoding="utf-8")
    return out_path


def _broadcast_daily_to_minutes(daily: pd.Series, minute_index: pd.MultiIndex) -> pd.Series:
    dates = minute_index.get_level_values(0)
    values = daily.reindex(dates).to_numpy()
    return pd.Series(values, index=minute_index)


def _find_project_root(start: Path) -> Path:
    for parent in start.parents:
        if (parent / "assets" / "MinutesIdx.h5").exists():
            return parent
    return start.parents[4]


def main() -> int:
    project_root = _find_project_root(Path(__file__).resolve())
    cfg = Config(
        data_path=project_root / "assets/MinutesIdx.h5",
        hdf_key="/d",
        target_col="IC",
        output_dir=project_root / "output/support_resistance",
        start=None,
        end=None,
        max_days=120,
        max_bars=2000,
        filename="kline.png",
        html_filename="kline.html",
        daily_filename="kline_daily.png",
        daily_html_filename="kline_daily.html",
        interactive_renderer="plotly",
        show_volume_zone=False,
        volume_zone_window=20,
        volume_zone_mult=1.0,
        show_poc=True,
        poc_window=240,
        poc_windows=(120, 240, 720),
        poc_bins=40,
        poc_shift=1,
        poc_groupby_level=None,
        show_daily_poc=True,
        daily_poc_window=60,
        daily_poc_windows=(60, 120),
        daily_poc_bins=40,
        daily_poc_shift=1,
        poc_price_col=None,
        poc_volume_col="volume",
        fib_min_high_mult=1.3,
        fib_min_high_points=0.0,
        use_minute_kline=True,
        export_daily_only=True,
        contract=None,
        price_col=None,
        price_cols=Config.price_cols,
        open_cols=Config.open_cols,
        high_cols=Config.high_cols,
        low_cols=Config.low_cols,
        close_cols=Config.close_cols,
        volume_cols=Config.volume_cols,
    )

    if cfg.use_minute_kline:
        raw = _load_raw_hdf(cfg.data_path, cfg.hdf_key)
        contract = cfg.contract or cfg.target_col
        raw = _unwrap_nested_frames(raw, contract)
        raw = _maybe_make_multiindex(raw)
        ohlc = _build_minute_ohlc(
            raw,
            open_cols=cfg.open_cols,
            high_cols=cfg.high_cols,
            low_cols=cfg.low_cols,
            close_cols=cfg.close_cols,
            volume_cols=cfg.volume_cols,
            price_col=cfg.price_col or cfg.target_col,
            price_cols=cfg.price_cols,
        )
        daily_ohlc = _daily_ohlc_from_minute(ohlc)
    else:
        target_raw = load_target_series(cfg.data_path, cfg.hdf_key, cfg.target_col)
        ohlc = _build_daily_ohlc(target_raw)
        daily_ohlc = ohlc
    ohlc = _apply_date_filter(ohlc, cfg.start, cfg.end)
    ohlc_png = ohlc
    ohlc_html = ohlc
    html_mult = 2
    if cfg.use_minute_kline:
        if cfg.max_bars > 0 and len(ohlc_png) > cfg.max_bars:
            ohlc_png = ohlc_png.iloc[-cfg.max_bars :]
        max_bars_html = int(cfg.max_bars) * int(html_mult) if int(cfg.max_bars) > 0 else 0
        if max_bars_html > 0 and len(ohlc_html) > max_bars_html:
            ohlc_html = ohlc_html.iloc[-max_bars_html:]
    else:
        if cfg.max_days > 0 and len(ohlc_png) > cfg.max_days:
            ohlc_png = ohlc_png.iloc[-cfg.max_days :]
        max_days_html = int(cfg.max_days) * int(html_mult) if int(cfg.max_days) > 0 else 0
        if max_days_html > 0 and len(ohlc_html) > max_days_html:
            ohlc_html = ohlc_html.iloc[-max_days_html:]

    daily_ohlc_filtered = _apply_date_filter(daily_ohlc, cfg.start, cfg.end)
    daily_ohlc_plot = daily_ohlc_filtered
    if cfg.max_days > 0 and len(daily_ohlc_plot) > cfg.max_days:
        daily_ohlc_plot = daily_ohlc_plot.iloc[-cfg.max_days :]

    # If Fib swing exists in a rolling calc window, make sure the swing's low/high/touch
    # dates are included in the displayed daily window (so they are visible on chart).
    fib_info = None
    if cfg.show_fib and not daily_ohlc_filtered.empty and int(cfg.fib_calc_days) != 0:
        fib_src = daily_ohlc_filtered
        if int(cfg.fib_calc_days) > 0 and len(fib_src) > int(cfg.fib_calc_days):
            fib_src = fib_src.iloc[-int(cfg.fib_calc_days) :]
        fib_info = fibonacci_swing_support(
            fib_src,
            lookaround_days=int(cfg.fib_lookaround_days),
            min_high_mult=float(cfg.fib_min_high_mult),
            min_high_points=float(cfg.fib_min_high_points),
            ratios=cfg.fib_ratios,
            first_ratio=float(cfg.fib_first_ratio),
        )
        if fib_info is not None and not daily_ohlc_plot.empty:
            need_start = min(
                pd.to_datetime(fib_info["low_date"]),
                pd.to_datetime(fib_info["high_date"]),
                pd.to_datetime(fib_info["touch_date"]),
            )
            if pd.to_datetime(daily_ohlc_plot.index.min()) > need_start:
                daily_ohlc_plot = daily_ohlc_filtered.loc[daily_ohlc_filtered.index >= need_start]

    out_path = plot_kline(
        ohlc_png,
        cfg.output_dir,
        cfg.filename,
        show_top_sr=cfg.show_top_sr,
        top_sr_n=cfg.top_sr_n,
        show_topk_poc=False,
        poc_topk=cfg.top_sr_n,
        show_poc=cfg.show_poc,
        poc_window=cfg.poc_window,
        poc_windows=cfg.poc_windows,
        poc_bins=cfg.poc_bins,
        poc_shift=cfg.poc_shift,
        poc_groupby_level=cfg.poc_groupby_level,
        show_daily_poc=cfg.show_daily_poc,
        daily_poc_window=cfg.daily_poc_window,
        daily_poc_windows=cfg.daily_poc_windows,
        daily_poc_bins=cfg.daily_poc_bins,
        daily_poc_shift=cfg.daily_poc_shift,
        poc_price_col=cfg.poc_price_col,
        poc_volume_col=cfg.poc_volume_col,
        show_fib=cfg.show_fib,
        fib_lookaround_days=cfg.fib_lookaround_days,
        fib_min_high_mult=cfg.fib_min_high_mult,
        fib_min_high_points=cfg.fib_min_high_points,
        fib_ratios=cfg.fib_ratios,
        fib_first_ratio=cfg.fib_first_ratio,
        daily_ohlc=daily_ohlc_filtered,
    )
    html_path = plot_kline_interactive(
        ohlc_html,
        cfg.output_dir,
        cfg.html_filename,
        renderer=cfg.interactive_renderer,
        show_volume_zone=cfg.show_volume_zone,
        volume_zone_window=cfg.volume_zone_window,
        volume_zone_mult=cfg.volume_zone_mult,
        show_top_sr=cfg.show_top_sr,
        top_sr_n=cfg.top_sr_n,
        show_topk_poc=False,
        poc_topk=cfg.top_sr_n,
        show_poc=cfg.show_poc,
        poc_window=cfg.poc_window,
        poc_windows=cfg.poc_windows,
        poc_bins=cfg.poc_bins,
        poc_shift=cfg.poc_shift,
        poc_groupby_level=cfg.poc_groupby_level,
        show_daily_poc=cfg.show_daily_poc,
        daily_poc_window=cfg.daily_poc_window,
        daily_poc_windows=cfg.daily_poc_windows,
        daily_poc_bins=cfg.daily_poc_bins,
        daily_poc_shift=cfg.daily_poc_shift,
        poc_price_col=cfg.poc_price_col,
        poc_volume_col=cfg.poc_volume_col,
        show_fib=cfg.show_fib,
        fib_lookaround_days=cfg.fib_lookaround_days,
        fib_min_high_mult=cfg.fib_min_high_mult,
        fib_min_high_points=cfg.fib_min_high_points,
        fib_ratios=cfg.fib_ratios,
        fib_first_ratio=cfg.fib_first_ratio,
        fib_daily_ohlc=daily_ohlc_filtered,
        daily_ohlc=daily_ohlc_filtered if isinstance(ohlc_html.index, pd.MultiIndex) else ohlc_html,
    )
    print(f"[ok] kline saved to: {out_path}")
    print(f"[ok] kline interactive saved to: {html_path}")

    if cfg.export_daily_only:
        daily_png = plot_kline(
            daily_ohlc_plot,
            cfg.output_dir,
            cfg.daily_filename,
            show_poc=False,
            show_daily_poc=True,
            daily_poc_window=cfg.daily_poc_window,
            daily_poc_windows=cfg.daily_poc_windows,
            daily_poc_bins=cfg.daily_poc_bins,
            daily_poc_shift=cfg.daily_poc_shift,
            poc_price_col=cfg.poc_price_col,
            poc_volume_col=cfg.poc_volume_col,
            show_fib=cfg.show_fib,
            fib_lookaround_days=cfg.fib_lookaround_days,
            fib_min_high_mult=cfg.fib_min_high_mult,
            fib_min_high_points=cfg.fib_min_high_points,
            fib_ratios=cfg.fib_ratios,
            fib_first_ratio=cfg.fib_first_ratio,
            daily_ohlc=daily_ohlc_plot,
        )
        daily_html = plot_kline_interactive(
            daily_ohlc_plot,
            cfg.output_dir,
            cfg.daily_html_filename,
            renderer=cfg.interactive_renderer,
            show_volume_zone=False,
            show_poc=False,
            show_daily_poc=True,
            daily_poc_window=cfg.daily_poc_window,
            daily_poc_windows=cfg.daily_poc_windows,
            daily_poc_bins=cfg.daily_poc_bins,
            daily_poc_shift=cfg.daily_poc_shift,
            poc_price_col=cfg.poc_price_col,
            poc_volume_col=cfg.poc_volume_col,
            show_fib=cfg.show_fib,
            fib_lookaround_days=cfg.fib_lookaround_days,
            fib_min_high_mult=cfg.fib_min_high_mult,
            fib_min_high_points=cfg.fib_min_high_points,
            fib_ratios=cfg.fib_ratios,
            fib_first_ratio=cfg.fib_first_ratio,
            fib_daily_ohlc=daily_ohlc_plot,
            daily_ohlc=daily_ohlc_plot,
        )
        print(f"[ok] daily kline saved to: {daily_png}")
        print(f"[ok] daily kline interactive saved to: {daily_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
