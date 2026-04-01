"""Simple z-score strategy."""

from __future__ import annotations

import numpy as np
import polars as pl

from ....strategy_types import BaseStrategy, StrategyContext


class SimpleZScoreStrategy(BaseStrategy):
    """Toy strategy that overweights lower MVRV z-score days."""

    strategy_id = "simple-zscore"
    version = "1.0.0"
    description = "Toy strategy that overweights lower mvrv_zscore days."

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


__all__ = ["SimpleZScoreStrategy"]
