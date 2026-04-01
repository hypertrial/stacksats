"""Momentum strategy."""

from __future__ import annotations

import numpy as np
import polars as pl

from ....strategy_types import BaseStrategy, StrategyContext


class MomentumStrategy(BaseStrategy):
    """Simple momentum strategy based on 30-day price trend."""

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


__all__ = ["MomentumStrategy"]
