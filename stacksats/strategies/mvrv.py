"""Built-in MVRV + MA strategy wrapper."""

from __future__ import annotations

import pandas as pd

from ..model_development import FEATS, compute_preference_scores
from ..strategy_types import BaseStrategy, StrategyContext


class MVRVStrategy(BaseStrategy):
    """Default strategy backed by `model_development.compute_window_weights`."""

    strategy_id = "mvrv"
    version = "1.0.0"
    description = "Built-in MVRV + MA allocation strategy."

    def required_feature_columns(self) -> tuple[str, ...]:
        return tuple(FEATS)

    def build_target_profile(
        self,
        ctx: StrategyContext,
        features_df: pd.DataFrame,
        signals: dict[str, pd.Series],
    ) -> pd.Series:
        del signals
        return compute_preference_scores(
            features_df=features_df,
            start_date=ctx.start_date,
            end_date=ctx.end_date,
        )
