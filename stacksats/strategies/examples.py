"""Example strategies for user experimentation."""

from __future__ import annotations

import numpy as np
import pandas as pd

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
        features_df: pd.DataFrame,
        signals: dict[str, pd.Series],
    ) -> pd.Series:
        del ctx, signals
        window = features_df
        if window.empty:
            return pd.Series(dtype=float)
        z = window.get("mvrv_zscore", pd.Series(0.0, index=window.index)).to_numpy()
        return pd.Series(-z, index=window.index)


class MomentumStrategy(BaseStrategy):
    """Simple momentum strategy based on 30-day price trend.

    Days with weaker short-term momentum receive slightly more allocation
    (contrarian tilt), while stronger momentum days get less.
    """

    strategy_id = "momentum"
    version = "1.0.0"
    description = "Simple momentum strategy with contrarian tilt."

    def required_feature_columns(self) -> tuple[str, ...]:
        return ("PriceUSD_coinmetrics",)

    def build_target_profile(
        self,
        ctx: StrategyContext,
        features_df: pd.DataFrame,
        signals: dict[str, pd.Series],
    ) -> pd.Series:
        del ctx, signals
        window = features_df
        if window.empty:
            return pd.Series(dtype=float)
        price = window["PriceUSD_coinmetrics"]
        momentum = price.pct_change(30).fillna(0.0)
        return pd.Series(-np.clip(momentum.to_numpy(), -1.0, 1.0), index=window.index)
