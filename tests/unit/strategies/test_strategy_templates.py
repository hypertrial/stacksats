from __future__ import annotations

import numpy as np

from stacksats.strategies.templates.minimal_profile import MinimalProfileTemplateStrategy
from stacksats.strategies.templates.minimal_propose import MinimalProposeTemplateStrategy
from stacksats.strategy_types import DayState, strategy_context_from_features_df
from tests.test_helpers import btc_frame


def _template_features_df():
    return btc_frame(
        start="2024-01-01",
        days=5,
        price_start=20000.0,
        price_step=100.0,
    ).with_columns(mvrv=np.linspace(0.8, 1.2, 5))


def test_minimal_profile_template_builds_placeholder_profile() -> None:
    features_df = _template_features_df()
    ctx = strategy_context_from_features_df(
        features_df,
        features_df["date"][0],
        features_df["date"][-1],
        features_df["date"][-1],
    )
    strategy = MinimalProfileTemplateStrategy()

    transformed = strategy.transform_features(ctx)
    signals = strategy.build_signals(ctx, transformed)
    profile = strategy.build_target_profile(ctx, transformed, signals)

    assert strategy.required_feature_sets() == ("core_model_features_v1",)
    assert strategy.required_feature_columns() == ()
    assert transformed.equals(features_df)
    assert signals["placeholder"].to_list() == [0.0] * features_df.height
    assert profile.mode == "preference"
    assert profile.values["date"].to_list() == features_df["date"].to_list()
    assert profile.values["value"].to_list() == [0.0] * features_df.height


def test_minimal_propose_template_returns_uniform_weight() -> None:
    features_df = _template_features_df()
    strategy = MinimalProposeTemplateStrategy()
    ctx = strategy_context_from_features_df(
        features_df,
        features_df["date"][0],
        features_df["date"][-1],
        features_df["date"][-1],
    )
    state = DayState(
        current_date=features_df["date"][0],
        features=features_df.head(1),
        remaining_budget=1.0,
        day_index=0,
        total_days=8,
        uniform_weight=0.125,
    )

    assert strategy.required_feature_sets() == ("core_model_features_v1",)
    assert strategy.required_feature_columns() == ()
    assert strategy.transform_features(ctx).equals(features_df)
    assert strategy.propose_weight(state) == 0.125
