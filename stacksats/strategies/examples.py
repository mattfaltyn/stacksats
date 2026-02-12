"""Example strategies for user experimentation."""

from __future__ import annotations

import numpy as np
import pandas as pd


class UniformStrategy:
    """Baseline strategy that allocates equally across the full date window."""

    def compute_weights(
        self,
        features_df: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        current_date: pd.Timestamp,
    ) -> pd.Series:
        del current_date  # interface compatibility
        window = features_df.loc[start_date:end_date]
        if window.empty:
            return pd.Series(dtype=float)
        n = len(window.index)
        return pd.Series(np.full(n, 1.0 / n), index=window.index)


class SimpleZScoreStrategy:
    """Toy strategy that overweights lower MVRV z-score days.

    A lower MVRV z-score implies relative undervaluation. This strategy converts
    that to a positive signal via ``exp(-zscore)`` and normalizes to sum to 1.
    """

    def compute_weights(
        self,
        features_df: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        current_date: pd.Timestamp,
    ) -> pd.Series:
        del current_date  # interface compatibility
        window = features_df.loc[start_date:end_date]
        if window.empty:
            return pd.Series(dtype=float)
        z = window.get("mvrv_zscore", pd.Series(0.0, index=window.index)).to_numpy()
        raw = np.exp(-z)
        weights = raw / raw.sum()
        return pd.Series(weights, index=window.index)


class MomentumStrategy:
    """Simple momentum strategy based on 30-day price trend.

    Days with weaker short-term momentum receive slightly more allocation
    (contrarian tilt), while stronger momentum days get less.
    """

    def compute_weights(
        self,
        features_df: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        current_date: pd.Timestamp,
    ) -> pd.Series:
        del current_date  # interface compatibility
        window = features_df.loc[start_date:end_date]
        if window.empty:
            return pd.Series(dtype=float)
        price = window["PriceUSD_coinmetrics"]
        momentum = price.pct_change(30).fillna(0.0)
        raw = np.exp(-np.clip(momentum.to_numpy(), -1.0, 1.0))
        weights = raw / raw.sum()
        return pd.Series(weights, index=window.index)
