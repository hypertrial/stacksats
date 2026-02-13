from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stacksats.model_development import (
    allocate_from_proposals,
    compute_mean_reversion_pressure,
    compute_preference_scores,
    compute_weights_from_proposals,
    compute_weights_from_target_profile,
)
from stacksats.prelude import compute_cycle_spd
from stacksats.runner import StrategyRunner
from stacksats.strategy_types import (
    BaseStrategy,
    StrategyContext,
    TargetProfile,
    ValidationConfig,
)


def _btc_df(days: int = 1600) -> pd.DataFrame:
    idx = pd.date_range("2021-01-01", periods=days, freq="D")
    return pd.DataFrame(
        {
            "PriceUSD_coinmetrics": np.linspace(10000.0, 60000.0, len(idx)),
            "CapMVRVCur": np.linspace(0.8, 2.2, len(idx)),
        },
        index=idx,
    )


class _NoIntentStrategy(BaseStrategy):
    pass


class _LeakyProfileStrategy(BaseStrategy):
    strategy_id = "leaky-profile"

    def transform_features(self, ctx: StrategyContext) -> pd.DataFrame:
        window = ctx.features_df.loc[ctx.start_date : ctx.end_date].copy()
        leak_value = float(ctx.features_df["PriceUSD_coinmetrics"].mean(skipna=True))
        window["leak"] = leak_value
        return window

    def build_target_profile(
        self,
        ctx: StrategyContext,
        features_df: pd.DataFrame,
        signals: dict[str, pd.Series],
    ) -> TargetProfile:
        del ctx, signals
        if features_df.empty:
            return TargetProfile(values=pd.Series(dtype=float), mode="absolute")
        values = pd.Series(-1.0, index=features_df.index, dtype=float)
        values.iloc[-1] = -float(features_df["leak"].iloc[-1])
        return TargetProfile(values=values, mode="absolute")


def test_runner_contract_rejects_strategy_without_intent_hooks() -> None:
    runner = StrategyRunner()
    with pytest.raises(TypeError, match="must implement propose_weight"):
        runner._validate_strategy_contract(_NoIntentStrategy())


def test_runner_validate_weights_accepts_empty_series() -> None:
    runner = StrategyRunner()
    runner._validate_weights(
        pd.Series(dtype=float),
        window_start=pd.Timestamp("2024-01-01"),
        window_end=pd.Timestamp("2024-01-02"),
    )


def test_runner_validate_detects_profile_only_forward_leakage() -> None:
    runner = StrategyRunner()
    result = runner.validate(
        _LeakyProfileStrategy(),
        ValidationConfig(
            start_date="2022-01-01",
            end_date="2024-12-31",
            min_win_rate=0.0,
        ),
        btc_df=_btc_df(),
    )

    assert bool(result.forward_leakage_ok) is False
    assert bool(result.passed) is False
    assert any("Forward leakage detected near" in message for message in result.messages)


def _single_window_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = pd.date_range("2024-01-01", "2025-01-01", freq="D")
    btc_df = pd.DataFrame(
        {"PriceUSD_coinmetrics": np.linspace(30000.0, 50000.0, len(idx))},
        index=idx,
    )
    features_df = pd.DataFrame(
        {"PriceUSD_coinmetrics": btc_df["PriceUSD_coinmetrics"]},
        index=idx,
    )
    return btc_df, features_df


def test_compute_cycle_spd_falls_back_to_uniform_for_empty_weights() -> None:
    btc_df, features_df = _single_window_data()
    result = compute_cycle_spd(
        btc_df,
        strategy_function=lambda _window: pd.Series(dtype=float),
        features_df=features_df,
        start_date="2024-01-01",
        end_date="2025-01-01",
        validate_weights=True,
    )

    row = result.iloc[0]
    assert row["dynamic_sats_per_dollar"] == pytest.approx(row["uniform_sats_per_dollar"])


def test_compute_cycle_spd_falls_back_to_uniform_for_nonfinite_weights() -> None:
    btc_df, features_df = _single_window_data()
    result = compute_cycle_spd(
        btc_df,
        strategy_function=lambda window: pd.Series(np.inf, index=window.index, dtype=float),
        features_df=features_df,
        start_date="2024-01-01",
        end_date="2025-01-01",
        validate_weights=True,
    )

    row = result.iloc[0]
    assert row["dynamic_sats_per_dollar"] == pytest.approx(row["uniform_sats_per_dollar"])


def test_compute_cycle_spd_raises_when_weight_validation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    btc_df, features_df = _single_window_data()

    class _NumpyProxy:
        @staticmethod
        def isclose(*_args, **_kwargs):
            return False

        def __getattr__(self, name):
            return getattr(np, name)

    monkeypatch.setattr("stacksats.prelude.np", _NumpyProxy())

    with pytest.raises(ValueError, match="sum to"):
        compute_cycle_spd(
            btc_df,
            strategy_function=lambda window: pd.Series(
                np.ones(len(window), dtype=float), index=window.index
            ),
            features_df=features_df,
            start_date="2024-01-01",
            end_date="2025-01-01",
            validate_weights=True,
        )


def test_compute_mean_reversion_pressure_adds_extreme_term() -> None:
    zscores = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=float)
    pressure = compute_mean_reversion_pressure(zscores)
    baseline = np.tanh(zscores * 0.5)

    assert pressure[0] < baseline[0]
    assert pressure[-1] > baseline[-1]
    assert np.all(np.abs(pressure) <= 1.0 + 1e-12)


def test_allocate_from_proposals_returns_empty_when_total_is_zero() -> None:
    weights = allocate_from_proposals(np.array([], dtype=float), n_past=0, n_total=0)
    assert weights.size == 0


def test_allocate_from_proposals_returns_uniform_when_no_past_days() -> None:
    weights = allocate_from_proposals(
        np.array([0.9, 0.1, 0.0], dtype=float),
        n_past=0,
        n_total=3,
    )
    np.testing.assert_allclose(weights, np.full(3, 1.0 / 3.0))


def test_compute_preference_scores_handles_missing_optional_features() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    features_df = pd.DataFrame(
        {
            "price_vs_ma": [0.1, -0.1, 0.2, -0.2],
            "mvrv_zscore": [0.5, -0.5, 1.0, -1.0],
            "mvrv_gradient": [0.2, -0.2, 0.3, -0.3],
        },
        index=idx,
    )

    scores = compute_preference_scores(features_df, idx.min(), idx.max())
    assert scores.index.equals(idx)
    assert np.isfinite(scores.to_numpy(dtype=float)).all()


def test_compute_weights_from_target_profile_returns_empty_for_empty_range() -> None:
    result = compute_weights_from_target_profile(
        features_df=pd.DataFrame(),
        start_date=pd.Timestamp("2024-01-02"),
        end_date=pd.Timestamp("2024-01-01"),
        current_date=pd.Timestamp("2024-01-01"),
        target_profile=pd.Series(dtype=float),
        mode="preference",
    )
    assert result.empty


def test_compute_weights_from_target_profile_absolute_uses_base_when_all_nonpositive() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    target_profile = pd.Series([-5.0, 0.0, -1.0, np.nan], index=idx, dtype=float)

    weights = compute_weights_from_target_profile(
        features_df=pd.DataFrame(index=idx),
        start_date=idx.min(),
        end_date=idx.max(),
        current_date=idx.max(),
        target_profile=target_profile,
        mode="absolute",
    )

    np.testing.assert_allclose(weights.to_numpy(dtype=float), np.full(len(idx), 1.0 / len(idx)))


def test_compute_weights_from_target_profile_rejects_unsupported_mode() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    with pytest.raises(ValueError, match="Unsupported target profile mode"):
        compute_weights_from_target_profile(
            features_df=pd.DataFrame(index=idx),
            start_date=idx.min(),
            end_date=idx.max(),
            current_date=idx.max(),
            target_profile=pd.Series([0.0, 0.0, 0.0], index=idx, dtype=float),
            mode="unknown",
        )


def test_compute_weights_from_proposals_returns_empty_for_empty_range() -> None:
    result = compute_weights_from_proposals(
        proposals=pd.Series(dtype=float),
        start_date=pd.Timestamp("2024-01-02"),
        end_date=pd.Timestamp("2024-01-01"),
        n_past=0,
    )
    assert result.empty


class _UniformProposalStrategy(BaseStrategy):
    strategy_id = "uniform-proposal"

    def propose_weight(self, state) -> float:
        return state.uniform_weight


class _NanProposalStrategy(BaseStrategy):
    strategy_id = "nan-proposal"

    def propose_weight(self, state) -> float:
        del state
        return float("nan")


def _strategy_context(periods: int = 3) -> StrategyContext:
    idx = pd.date_range("2024-01-01", periods=periods, freq="D")
    features_df = pd.DataFrame(
        {"PriceUSD_coinmetrics": np.linspace(100.0, 110.0, len(idx))},
        index=idx,
    )
    return StrategyContext(
        features_df=features_df,
        start_date=idx.min(),
        end_date=idx.max(),
        current_date=idx.max(),
    )


def test_base_strategy_default_build_target_profile_returns_absolute_profile() -> None:
    strategy = _UniformProposalStrategy()
    ctx = _strategy_context(periods=4)
    features_df = strategy.transform_features(ctx)

    profile = strategy.build_target_profile(ctx, features_df, signals={})

    assert isinstance(profile, TargetProfile)
    assert profile.mode == "absolute"
    assert len(profile.values) == 4
    assert np.isfinite(profile.values.to_numpy(dtype=float)).all()


def test_base_strategy_default_build_target_profile_handles_empty_features() -> None:
    strategy = _UniformProposalStrategy()
    ctx = _strategy_context(periods=3)
    profile = strategy.build_target_profile(ctx, pd.DataFrame(), signals={})

    assert profile.mode == "absolute"
    assert profile.values.empty


def test_base_strategy_default_build_target_profile_rejects_nonfinite_proposals() -> None:
    strategy = _NanProposalStrategy()
    ctx = _strategy_context(periods=3)
    features_df = strategy.transform_features(ctx)

    with pytest.raises(ValueError, match="finite numeric value"):
        strategy.build_target_profile(ctx, features_df, signals={})


def test_base_strategy_validate_weights_allows_empty_weights() -> None:
    strategy = _UniformProposalStrategy()
    strategy.validate_weights(pd.Series(dtype=float), _strategy_context())
