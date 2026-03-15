from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from stacksats.runner import StrategyRunner
from stacksats.strategy_types import BaseStrategy, StrategyContext, ValidationConfig
from tests.test_helpers import btc_frame


def _window_dates(ctx: StrategyContext) -> list[dt.datetime]:
    return pl.datetime_range(ctx.start_date, ctx.end_date, interval="1d", eager=True).to_list()


class UniformProposeStrategy(BaseStrategy):
    strategy_id = "runner-uniform"
    version = "1.0.0"

    def propose_weight(self, state):
        return state.uniform_weight


class MutatingProposeStrategy(BaseStrategy):
    strategy_id = "runner-mutating"
    version = "1.0.0"

    def transform_features(self, ctx: StrategyContext) -> pl.DataFrame:
        ctx.features_df.insert_column(
            ctx.features_df.width,
            pl.Series("__mutated__", [1.0] * ctx.features_df.height),
        )
        return ctx.features_df.filter(
            (pl.col("date") >= ctx.start_date) & (pl.col("date") <= ctx.end_date)
        )

    def propose_weight(self, state):
        return state.uniform_weight


class RandomProposeStrategy(BaseStrategy):
    strategy_id = "runner-random"
    version = "1.0.0"

    def propose_weight(self, state):
        rng = np.random.default_rng()
        return float(rng.uniform(0.0, state.uniform_weight * 2.0))


class ProfileOffsetLeakStrategy(BaseStrategy):
    strategy_id = "runner-leak-profile-offset"
    version = "1.0.0"

    def build_target_profile(
        self,
        ctx: StrategyContext,
        features_df: pl.DataFrame,
        signals: dict[str, pl.Series],
    ) -> pl.DataFrame:
        del signals
        future_rows = ctx.features_df.filter(pl.col("date") > ctx.end_date)
        offset = float(future_rows.height)
        base = np.linspace(-1.0, 1.0, features_df.height, dtype=float)
        return pl.DataFrame(
            {
                "date": features_df["date"],
                "value": base + offset,
            }
        )


class ProfileMutationStrategy(BaseStrategy):
    strategy_id = "runner-profile-mutation"
    version = "1.0.0"

    def build_target_profile(
        self,
        ctx: StrategyContext,
        features_df: pl.DataFrame,
        signals: dict[str, pl.Series],
    ) -> pl.DataFrame:
        del signals
        ctx.features_df.insert_column(
            ctx.features_df.width,
            pl.Series("__profile_mutation__", [1.0] * ctx.features_df.height),
        )
        return pl.DataFrame(
            {
                "date": features_df["date"],
                "value": [1.0] * features_df.height,
            }
        )


class DualHookProfilePreferredLeakStrategy(BaseStrategy):
    strategy_id = "runner-dual-profile-leak"
    version = "1.0.0"
    intent_preference = "profile"

    def propose_weight(self, state):
        return state.uniform_weight

    def build_target_profile(
        self,
        ctx: StrategyContext,
        features_df: pl.DataFrame,
        signals: dict[str, pl.Series],
    ) -> pl.DataFrame:
        del signals
        future_rows = ctx.features_df.filter(pl.col("date") > ctx.end_date)
        offset = float(future_rows.height)
        base = np.linspace(-1.0, 1.0, features_df.height, dtype=float)
        return pl.DataFrame(
            {
                "date": features_df["date"],
                "value": base + offset,
            }
        )


def btc_df(days: int = 900) -> pl.DataFrame:
    return btc_frame(start="2021-01-01", days=days)


def fast_strict_validation_config(**overrides) -> ValidationConfig:
    params = {
        "start_date": "2022-01-01",
        "end_date": "2024-03-31",
        "min_win_rate": 0.0,
        "strict": True,
        "bootstrap_trials": 25,
        "permutation_trials": 25,
        "block_size": 14,
    }
    params.update(overrides)
    return ValidationConfig(**params)


def patch_skip_weight_and_lock_checks(
    monkeypatch: pytest.MonkeyPatch,
    runner: StrategyRunner,
) -> None:
    monkeypatch.setattr(
        runner,
        "_run_weight_constraint_checks",
        lambda **kwargs: kwargs["end_ts"],
    )
    monkeypatch.setattr(runner, "_run_locked_prefix_check", lambda **kwargs: None)
