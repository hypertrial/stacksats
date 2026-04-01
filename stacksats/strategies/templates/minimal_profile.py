"""Copyable profile-mode strategy template.

This module is a research starter, not a built-in catalog entry.
"""

from __future__ import annotations

import polars as pl

from ...strategy_types import BaseStrategy, StrategyContext, TargetProfile


class MinimalProfileTemplateStrategy(BaseStrategy):
    """Minimal profile template for custom experiments."""

    strategy_id = "minimal-profile-template"
    version = "0.1.0"
    description = "Copyable profile-mode template for local strategy research."

    def required_feature_sets(self) -> tuple[str, ...]:
        return ("core_model_features_v1",)

    def required_feature_columns(self) -> tuple[str, ...]:
        return ()

    def transform_features(self, ctx: StrategyContext) -> pl.DataFrame:
        return ctx.features_df.clone()

    def build_signals(
        self,
        ctx: StrategyContext,
        features_df: pl.DataFrame,
    ) -> dict[str, pl.Series]:
        del ctx
        return {
            "placeholder": pl.Series(
                "placeholder",
                [0.0] * features_df.height,
                dtype=pl.Float64,
            ),
        }

    def build_target_profile(
        self,
        ctx: StrategyContext,
        features_df: pl.DataFrame,
        signals: dict[str, pl.Series],
    ) -> TargetProfile:
        del ctx
        return TargetProfile(
            values=pl.DataFrame(
                {
                    "date": features_df["date"],
                    "value": signals["placeholder"],
                }
            ),
            mode="preference",
        )


__all__ = ["MinimalProfileTemplateStrategy"]
