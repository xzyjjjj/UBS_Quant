from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_research.backtest.engine import get_dates, slice_day


@dataclass
class FeatureHandler:
    def get_feature_config(self) -> dict:
        return {
            "features": ["T_last_close", "T_vwap"],
            "signals": ["default_st"],
        }

    def build_features(self, target_raw: pd.DataFrame) -> pd.DataFrame:
        """Compute daily snapshot features and default signal."""
        dates = get_dates(target_raw)
        readouts = pd.DataFrame(index=dates)

        last_close = []
        vwap = []
        for day in dates:
            day_df = slice_day(target_raw, day)
            last_close.append(day_df["Close"].iloc[-1])
            vwap.append((day_df["Close"] * day_df["Volume"]).sum() / day_df["Volume"].sum())

        readouts["T_last_close"] = last_close
        readouts["T_vwap"] = vwap
        ratio = np.array(readouts["T_last_close"]) / np.array(readouts["T_vwap"])
        readouts["default_st"] = np.where(ratio > 1.0, -1, 1)
        return readouts
