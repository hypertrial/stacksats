"""Example strategies for user experimentation."""

from __future__ import annotations

import numpy as np
import polars as pl

from ..strategy_types import BaseStrategy, DayState, StrategyContext


class UniformStrategy(BaseStrategy):
    """Baseline strategy that allocates equally across the full date window."""

    strategy_id = "uniform"
    version = "1.0.0"
    description = "Uniform baseline strategy."

    def required_feature_columns(self) -> tuple[str, ...]:
        return ()

    def propose_weight(self, state: DayState) -> float:
        # Framework enforces clipping, remaining budget, and lock semantics.
        return float(state.uniform_weight)


class SimpleZScoreStrategy(BaseStrategy):
    """Toy strategy that overweights lower MVRV z-score days.

    A lower MVRV z-score implies relative undervaluation. This strategy converts
    that to a positive signal via ``exp(-zscore)`` and normalizes to sum to 1.
    """

    strategy_id = "simple-zscore"
    version = "1.0.0"
    description = "Toy strategy that overweights lower MVRV z-score days."

    def required_feature_columns(self) -> tuple[str, ...]:
        return ()

    def build_target_profile(
        self,
        ctx: StrategyContext,
        features_df: pl.DataFrame,
        signals: dict[str, pl.Series],
    ) -> pl.DataFrame:
        del ctx, signals
        if features_df.is_empty():
            return pl.DataFrame(schema={"date": pl.Datetime("us"), "value": pl.Float64})
        if "mvrv_zscore" in features_df.columns:
            z = features_df["mvrv_zscore"].to_numpy()
        else:
            z = np.zeros(features_df.height)
        values = -z
        return pl.DataFrame({
            "date": features_df["date"],
            "value": values,
        })


class MomentumStrategy(BaseStrategy):
    """Simple momentum strategy based on 30-day price trend.

    Days with weaker short-term momentum receive slightly more allocation
    (contrarian tilt), while stronger momentum days get less.
    """

    strategy_id = "momentum"
    version = "1.0.0"
    description = "Simple momentum strategy with contrarian tilt."

    def required_feature_columns(self) -> tuple[str, ...]:
        return ("price_usd",)

    def build_target_profile(
        self,
        ctx: StrategyContext,
        features_df: pl.DataFrame,
        signals: dict[str, pl.Series],
    ) -> pl.DataFrame:
        del ctx, signals
        if features_df.is_empty():
            return pl.DataFrame(schema={"date": pl.Datetime("us"), "value": pl.Float64})
        price = features_df["price_usd"]
        momentum = price.pct_change(30).fill_null(0.0)
        values = -np.clip(momentum.to_numpy(), -1.0, 1.0)
        return pl.DataFrame({
            "date": features_df["date"],
            "value": values,
        })
