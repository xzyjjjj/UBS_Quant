from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd

from quant_research.backtest.engine import get_dates, slice_day
from quant_research.alphas import alpha_basic  # register default alphas
from quant_research.alphas.registry import get_alpha
from quant_research.features.feature101 import (
    alpha_001,
    alpha_002,
    alpha_003,
    alpha_004,
    alpha_005,
    alpha_006,
    alpha_007,
    alpha_008,
    alpha_009,
    alpha_010,
)
from quant_research.features.support_resistance import build_support_resistance_features


@dataclass
class FeatureHandler:
    alpha_list: Tuple[str, ...] = ("default_st",)

    def get_feature_config(self) -> dict:
        return {
            "features": [
                "T_last_close",
                "T_vwap",
                "feature_101",
                "sr_support_20",
                "sr_resistance_20",
                "sr_support_bounce_long_20",
                "sr_resistance_breakout_long_20",
                "sr_support_bias_long_20",
                "sr_volume_node_20",
                "sr_volume_zone_low_20",
                "sr_volume_zone_high_20",
                "sr_volume_zone_bounce_long_20",
                "sr_sma_20",
                "sr_ema_20",
                "sr_vwap_20",
                "sr_sma_support_long_20",
                "sr_vwap_support_long_20",
                "sr_channel_mid_20",
                "sr_channel_upper_20",
                "sr_channel_lower_20",
                "sr_channel_breakout_long_20",
                "sr_channel_revert_long_20",
                "sr_fib_236_20",
                "sr_fib_382_20",
                "sr_fib_500_20",
                "sr_fib_618_20",
                "sr_fib_bounce_long_20",
                "alpha_001",
                "alpha_002",
                "alpha_003",
                "alpha_004",
                "alpha_005",
                "alpha_006",
                "alpha_007",
                "alpha_008",
                "alpha_009",
                "alpha_010",
            ],
            "alphas": list(self.alpha_list),
        }

    def build_features(self, target_raw: pd.DataFrame) -> pd.DataFrame:
        """Compute daily snapshot features and alphas."""
        dates = get_dates(target_raw)
        readouts = pd.DataFrame(index=dates)

        opens = []
        highs = []
        lows = []
        closes = []
        volumes = []
        vwap = []
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

            closes.append(float(close_series.iloc[-1]))
            opens.append(float(open_series.iloc[0]))
            highs.append(float(high_series.max()))
            lows.append(float(low_series.min()))
            if volume_series is None:
                volumes.append(float("nan"))
                vwap.append(float(close_series.iloc[-1]))
            else:
                volume_sum = float(volume_series.sum())
                volumes.append(volume_sum)
                if volume_sum == 0:
                    vwap.append(float(close_series.iloc[-1]))
                else:
                    vwap.append(float((close_series * volume_series).sum() / volume_sum))

        readouts["open"] = opens
        readouts["high"] = highs
        readouts["low"] = lows
        readouts["close"] = closes
        readouts["volume"] = volumes
        readouts["vwap"] = vwap
        readouts["returns"] = readouts["close"].pct_change()
        readouts["adv20"] = readouts["volume"].rolling(20).mean()

        readouts["T_last_close"] = readouts["close"]
        readouts["T_vwap"] = readouts["vwap"]

        feature_101_series = alpha_001(readouts)
        readouts["feature_101"] = feature_101_series
        readouts["alpha_001"] = feature_101_series
        readouts["alpha_002"] = alpha_002(readouts)
        readouts["alpha_003"] = alpha_003(readouts)
        readouts["alpha_004"] = alpha_004(readouts)
        readouts["alpha_005"] = alpha_005(readouts)
        readouts["alpha_006"] = alpha_006(readouts)
        readouts["alpha_007"] = alpha_007(readouts)
        readouts["alpha_008"] = alpha_008(readouts)
        readouts["alpha_009"] = alpha_009(readouts)
        readouts["alpha_010"] = alpha_010(readouts)
        readouts = build_support_resistance_features(readouts, windows=(20,))
        for alpha_name in self.alpha_list:
            alpha_fn = get_alpha(alpha_name)
            readouts[alpha_name] = alpha_fn(readouts)
        return readouts
