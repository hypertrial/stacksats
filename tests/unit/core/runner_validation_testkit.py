from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stacksats.runner import StrategyRunner
from stacksats.strategy_types import BaseStrategy, StrategyContext, ValidationConfig


class UniformProposeStrategy(BaseStrategy):
    strategy_id = "runner-uniform"
    version = "1.0.0"

    def propose_weight(self, state):
        return state.uniform_weight


class MutatingProposeStrategy(BaseStrategy):
    strategy_id = "runner-mutating"
    version = "1.0.0"

    def transform_features(self, ctx):
        # Intentional contract violation for strict-mode guard coverage.
        ctx.features_df["__mutated__"] = 1.0
        return ctx.features_df.loc[ctx.start_date : ctx.end_date].copy()

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
        features_df: pd.DataFrame,
        signals: dict[str, pd.Series],
    ) -> pd.Series:
        del signals
        future_rows = ctx.features_df.index[ctx.features_df.index > ctx.end_date]
        offset = float(len(future_rows))
        # Additive offsets leave softmax weights unchanged, but profile checks should still catch leakage.
        base = np.linspace(-1.0, 1.0, len(features_df), dtype=float)
        return pd.Series(base + offset, index=features_df.index, dtype=float)


class ProfileMutationStrategy(BaseStrategy):
    strategy_id = "runner-profile-mutation"
    version = "1.0.0"

    def build_target_profile(
        self,
        ctx: StrategyContext,
        features_df: pd.DataFrame,
        signals: dict[str, pd.Series],
    ) -> pd.Series:
        del signals
        ctx.features_df["__profile_mutation__"] = 1.0
        if features_df.empty:
            return pd.Series(dtype=float)
        return pd.Series(np.ones(len(features_df), dtype=float), index=features_df.index)


class DualHookProfilePreferredLeakStrategy(BaseStrategy):
    strategy_id = "runner-dual-profile-leak"
    version = "1.0.0"
    intent_preference = "profile"

    def propose_weight(self, state):
        return state.uniform_weight

    def build_target_profile(
        self,
        ctx: StrategyContext,
        features_df: pd.DataFrame,
        signals: dict[str, pd.Series],
    ) -> pd.Series:
        del signals
        future_rows = ctx.features_df.index[ctx.features_df.index > ctx.end_date]
        offset = float(len(future_rows))
        base = np.linspace(-1.0, 1.0, len(features_df), dtype=float)
        return pd.Series(base + offset, index=features_df.index, dtype=float)


def btc_df(days: int = 900) -> pd.DataFrame:
    idx = pd.date_range("2021-01-01", periods=days, freq="D")
    return pd.DataFrame(
        {
            "price_usd": np.linspace(10000.0, 50000.0, len(idx)),
            "mvrv": np.linspace(1.0, 2.0, len(idx)),
        },
        index=idx,
    )


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
