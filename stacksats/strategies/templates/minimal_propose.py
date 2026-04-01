"""Copyable propose-mode strategy template.

This module is a research starter, not a built-in catalog entry.
"""

from __future__ import annotations

import polars as pl

from ...strategy_types import BaseStrategy, DayState, StrategyContext


class MinimalProposeTemplateStrategy(BaseStrategy):
    """Minimal daily proposal template for custom experiments."""

    strategy_id = "minimal-propose-template"
    version = "0.1.0"
    description = "Copyable propose-mode template for local strategy research."

    def required_feature_sets(self) -> tuple[str, ...]:
        return ("core_model_features_v1",)

    def required_feature_columns(self) -> tuple[str, ...]:
        return ()

    def transform_features(self, ctx: StrategyContext) -> pl.DataFrame:
        return ctx.features_df.clone()

    def propose_weight(self, state: DayState) -> float:
        # Replace this placeholder with a causal daily proposal rule.
        return float(state.uniform_weight)


__all__ = ["MinimalProposeTemplateStrategy"]
