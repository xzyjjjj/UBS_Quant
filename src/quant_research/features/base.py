"""Feature abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class Feature(ABC):
    """Base feature interface."""

    name: str = "feature"

    @abstractmethod
    def compute(self, data: pd.DataFrame) -> pd.Series | pd.DataFrame:
        raise NotImplementedError


class TimeSeriesFeature(Feature):
    """Marker class for time-series features."""

    def compute(self, data: pd.DataFrame) -> pd.Series | pd.DataFrame:  # pragma: no cover - abstract
        raise NotImplementedError
