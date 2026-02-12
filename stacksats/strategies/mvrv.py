"""Built-in MVRV + MA strategy wrapper."""

from __future__ import annotations

import pandas as pd

from ..model_development import compute_window_weights


class MVRVStrategy:
    """Default strategy backed by `model_development.compute_window_weights`."""

    def compute_weights(
        self,
        features_df: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        current_date: pd.Timestamp,
    ) -> pd.Series:
        return compute_window_weights(features_df, start_date, end_date, current_date)
