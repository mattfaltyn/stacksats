"""Example strategy using feature/signal/target-profile hooks.

Users define transformed features and signal formulas.
Framework computes final iterative allocation weights.

This example uses the batch hook (`build_target_profile`).
For day-by-day control, strategies can alternatively implement `propose_weight(state)`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import pandas as pd

from stacksats import BacktestConfig, BaseStrategy, StrategyContext, TargetProfile, ValidationConfig


@dataclass(frozen=True)
class ExampleConfig:
    value_weight: float = 0.7
    trend_weight: float = 0.3
    strength: float = 4.0
    trend_lookback_days: int = 30


class ExampleMVRVStrategy(BaseStrategy):
    """Example strategy with user-owned signals and framework-owned allocation."""

    strategy_id = "example-mvrv"
    version = "2.0.0"
    description = "Example hook-based strategy where framework performs iteration."

    def __init__(
        self,
        value_weight: float = 0.7,
        trend_weight: float = 0.3,
        strength: float = 4.0,
        trend_lookback_days: int = 30,
    ):
        self.cfg = ExampleConfig(
            value_weight=value_weight,
            trend_weight=trend_weight,
            strength=strength,
            trend_lookback_days=trend_lookback_days,
        )

    def transform_features(self, ctx: StrategyContext) -> pd.DataFrame:
        window = ctx.features_df.loc[ctx.start_date : ctx.end_date].copy()
        if window.empty:
            return window
        price = window[ctx.btc_price_col]
        trend = price.pct_change(self.cfg.trend_lookback_days).fillna(0.0)
        window["trend_signal_input"] = trend.clip(-1.0, 1.0)
        return window

    def build_signals(
        self,
        ctx: StrategyContext,
        features_df: pd.DataFrame,
    ) -> dict[str, pd.Series]:
        del ctx
        value_signal = -features_df.get(
            "mvrv_zscore",
            pd.Series(0.0, index=features_df.index),
        ).clip(-4, 4)
        trend_signal = -features_df["trend_signal_input"]
        return {
            "value": value_signal,
            "trend": trend_signal,
        }

    def build_target_profile(
        self,
        ctx: StrategyContext,
        features_df: pd.DataFrame,
        signals: dict[str, pd.Series],
    ) -> TargetProfile:
        # Batch hook: return per-day preference intent; framework handles iteration.
        del ctx, features_df
        preference = (
            (self.cfg.value_weight * signals["value"])
            + (self.cfg.trend_weight * signals["trend"])
        ) * self.cfg.strength
        return TargetProfile(values=preference, mode="preference")


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
