"""Example strategy wired to StackSats model_development signal logic.

The strategy computes the same feature set and dynamic multipliers as
`stacksats.model_development`, then hands absolute target magnitudes to the
StackSats framework allocation kernel.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from stacksats import model_development as model_lib
from stacksats import (
    BacktestConfig,
    BaseStrategy,
    StrategyContext,
    TargetProfile,
    ValidationConfig,
)


class ExampleMVRVStrategy(BaseStrategy):
    """Model-equivalent strategy using StackSats production signal formulas."""

    strategy_id = "example-mvrv"
    version = "3.0.0"
    description = "Uses stacksats.model_development feature engineering and multiplier logic."

    @staticmethod
    def _clean_array(values: pd.Series) -> np.ndarray:
        arr = values.to_numpy(dtype=float)
        return np.where(np.isfinite(arr), arr, 0.0)

    def transform_features(self, ctx: StrategyContext) -> pd.DataFrame:
        # Runner already passes precomputed model features in ctx.features_df.
        # Recomputing here drops raw MVRV inputs and degrades parity with runtime export.
        return ctx.features_df.loc[ctx.start_date : ctx.end_date].copy()

    def build_signals(
        self,
        ctx: StrategyContext,
        features_df: pd.DataFrame,
    ) -> dict[str, pd.Series]:
        del ctx, features_df
        return {}

    def build_target_profile(
        self,
        ctx: StrategyContext,
        features_df: pd.DataFrame,
        signals: dict[str, pd.Series],
    ) -> TargetProfile:
        del ctx, signals
        if features_df.empty:
            return TargetProfile(values=pd.Series(dtype=float), mode="absolute")

        n = len(features_df.index)
        base = np.ones(n, dtype=float) / n

        price_vs_ma = self._clean_array(features_df["price_vs_ma"])
        mvrv_zscore = self._clean_array(features_df["mvrv_zscore"])
        mvrv_gradient = self._clean_array(features_df["mvrv_gradient"])

        if "mvrv_percentile" in features_df.columns:
            mvrv_percentile = self._clean_array(features_df["mvrv_percentile"])
            mvrv_percentile = np.where(mvrv_percentile == 0.0, 0.5, mvrv_percentile)
        else:
            mvrv_percentile = None

        if "mvrv_acceleration" in features_df.columns:
            mvrv_acceleration = self._clean_array(features_df["mvrv_acceleration"])
        else:
            mvrv_acceleration = None

        if "mvrv_volatility" in features_df.columns:
            mvrv_volatility = self._clean_array(features_df["mvrv_volatility"])
            mvrv_volatility = np.where(mvrv_volatility == 0.0, 0.5, mvrv_volatility)
        else:
            mvrv_volatility = None

        if "signal_confidence" in features_df.columns:
            signal_confidence = self._clean_array(features_df["signal_confidence"])
            signal_confidence = np.where(signal_confidence == 0.0, 0.5, signal_confidence)
        else:
            signal_confidence = None

        multiplier = model_lib.compute_dynamic_multiplier(
            price_vs_ma,
            mvrv_zscore,
            mvrv_gradient,
            mvrv_percentile,
            mvrv_acceleration,
            mvrv_volatility,
            signal_confidence,
        )
        raw = base * multiplier
        absolute = pd.Series(raw, index=features_df.index, dtype=float)
        return TargetProfile(values=absolute, mode="absolute")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hook-based StackSats example strategy.")
    parser.add_argument("--start-date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--strategy-label", type=str, default="example-mvrv")
    args = parser.parse_args()

    strategy = ExampleMVRVStrategy()
    validation = strategy.validate(
        ValidationConfig(start_date=args.start_date, end_date=args.end_date)
    )
    print(validation.summary())
    for message in validation.messages:
        print(f"- {message}")

    result = strategy.backtest(
        BacktestConfig(
            start_date=args.start_date,
            end_date=args.end_date,
            strategy_label=args.strategy_label,
        )
    )
    print(result.summary())
    result.plot(output_dir=args.output_dir)
    result.to_json(f"{args.output_dir}/backtest_result.json")


if __name__ == "__main__":
    main()
