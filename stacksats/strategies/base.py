"""Strategy interfaces for StackSats."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import pandas as pd


class WindowStrategy(Protocol):
    """Protocol for date-window weight strategies."""

    def compute_weights(
        self,
        features_df: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        current_date: pd.Timestamp,
    ) -> pd.Series:
        """Return weights indexed by dates for the provided window."""


@dataclass(slots=True)
class CallableWindowStrategy:
    """Adapt a plain callable into a `WindowStrategy`."""

    fn: Callable[[pd.DataFrame, pd.Timestamp, pd.Timestamp, pd.Timestamp], pd.Series]

    def compute_weights(
        self,
        features_df: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        current_date: pd.Timestamp,
    ) -> pd.Series:
        return self.fn(features_df, start_date, end_date, current_date)
