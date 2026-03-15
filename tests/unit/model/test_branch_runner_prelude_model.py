from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest
from types import SimpleNamespace

import stacksats.prelude as prelude_module
from stacksats.model_development import (
    allocate_from_proposals,
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
    strategy_context_from_features_df,
)


def _btc_df() -> pl.DataFrame:
    dates = pl.datetime_range(
        dt.datetime(2023, 1, 1),
        dt.datetime(2024, 12, 31),
        interval="1d",
        eager=True,
    )
    n = dates.len()
    return pl.DataFrame({
        "date": dates,
        "price_usd": np.linspace(10000.0, 60000.0, n),
        "mvrv": np.linspace(0.8, 2.2, n),
    })


class _NoIntentStrategy(BaseStrategy):
    pass


class _LeakyProfileStrategy(BaseStrategy):
    strategy_id = "leaky-profile"

    def transform_features(self, ctx: StrategyContext) -> pl.DataFrame:
        features = ctx.features.data
        window = features.filter(
            (pl.col("date") >= ctx.start_date) & (pl.col("date") <= ctx.end_date)
        )
        leak_value = float(features["price_usd"].mean())
        return window.with_columns(pl.lit(leak_value).alias("leak"))

    def build_target_profile(
        self,
        ctx: StrategyContext,
        features_df: pl.DataFrame,
        signals: dict[str, pl.Series],
    ) -> TargetProfile:
        del ctx, signals
        if features_df.is_empty():
            return TargetProfile(
                values=pl.DataFrame(schema={"date": pl.Datetime("us"), "value": pl.Float64}),
                mode="absolute",
            )
        last_leak = float(features_df["leak"][-1])
        values = [-1.0] * features_df.height
        values[-1] = -last_leak
        return TargetProfile(
            values=pl.DataFrame({
                "date": features_df["date"],
                "value": pl.Series("value", values),
            }),
            mode="absolute",
        )


def test_runner_contract_rejects_strategy_without_intent_hooks() -> None:
    runner = StrategyRunner()
    with pytest.raises(TypeError, match="must implement propose_weight"):
        runner._validate_strategy_contract(_NoIntentStrategy())


def test_runner_validate_weights_accepts_empty_series() -> None:
    runner = StrategyRunner()
    runner._validate_weights(
        pl.DataFrame(schema={"date": pl.Datetime("us"), "weight": pl.Float64}),
        window_start=dt.datetime(2024, 1, 1),
        window_end=dt.datetime(2024, 1, 2),
    )


def test_runner_validate_observed_only_profile_input_blocks_profile_peeking() -> None:
    """Validation should detect and reject strategies that peek at future data."""
    runner = StrategyRunner()
    runner.backtest = lambda *args, **kwargs: SimpleNamespace(win_rate=100.0)
    result = runner.validate(
        _LeakyProfileStrategy(),
        ValidationConfig(
            start_date="2024-01-01",
            end_date="2024-12-31",
            min_win_rate=0.0,
        ),
        btc_df=_btc_df(),
    )

    assert bool(result.forward_leakage_ok) is False
    assert bool(result.passed) is False
    assert any("Forward leakage detected" in message for message in result.messages)


def _single_window_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    dates = pl.datetime_range(
        dt.datetime(2024, 1, 1),
        dt.datetime(2025, 1, 1),
        interval="1d",
        eager=True,
    )
    n = dates.len()
    btc_df = pl.DataFrame({
        "date": dates,
        "price_usd": np.linspace(30000.0, 50000.0, n),
    })
    features_df = pl.DataFrame({
        "date": dates,
        "price_usd": np.linspace(30000.0, 50000.0, n),
    })
    return btc_df, features_df


def _empty_weights_df() -> pl.DataFrame:
    return pl.DataFrame(schema={"date": pl.Datetime("us"), "weight": pl.Float64})


def test_compute_cycle_spd_falls_back_to_uniform_for_empty_weights() -> None:
    btc_df, features_df = _single_window_data()
    result = compute_cycle_spd(
        btc_df,
        strategy_function=lambda _window: _empty_weights_df(),
        features_df=features_df,
        start_date="2024-01-01",
        end_date="2025-01-01",
        validate_weights=True,
    )

    row = result.row(0, named=True)
    assert row["dynamic_sats_per_dollar"] == pytest.approx(row["uniform_sats_per_dollar"])


def test_compute_cycle_spd_falls_back_to_uniform_for_nonfinite_weights() -> None:
    btc_df, features_df = _single_window_data()
    result = compute_cycle_spd(
        btc_df,
        strategy_function=lambda window: pl.DataFrame({
            "date": window["date"],
            "weight": [float("inf")] * window.height,
        }),
        features_df=features_df,
        start_date="2024-01-01",
        end_date="2025-01-01",
        validate_weights=True,
    )

    row = result.row(0, named=True)
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
            strategy_function=lambda window: pl.DataFrame({
                "date": window["date"],
                "weight": [1.0] * window.height,
            }),
            features_df=features_df,
            start_date="2024-01-01",
            end_date="2025-01-01",
            validate_weights=True,
        )


def test_compute_cycle_spd_computes_features_when_none_provided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    btc_df, _ = _single_window_data()
    calls = {"count": 0}

    def _fake_precompute_features(df: pl.DataFrame) -> pl.DataFrame:
        calls["count"] += 1
        return df.select(["date", "price_usd"])

    monkeypatch.setattr("stacksats.prelude.precompute_features", _fake_precompute_features)

    result = compute_cycle_spd(
        btc_df,
        strategy_function=lambda window: pl.DataFrame({
            "date": window["date"],
            "weight": [1.0 / window.height] * window.height,
        }),
        features_df=None,
        start_date="2024-01-01",
        end_date="2025-01-01",
        validate_weights=True,
    )

    assert calls["count"] == 1
    assert result.height >= 1


def test_compute_cycle_spd_skips_window_when_window_end_exceeds_requested_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start_ts = dt.datetime(2024, 1, 1)
    end_ts = start_ts + prelude_module.WINDOW_OFFSET + dt.timedelta(days=2)
    full_dates = pl.datetime_range(
        start_ts,
        end_ts + dt.timedelta(days=2),
        interval="1d",
        eager=True,
    )
    btc_df = pl.DataFrame({
        "date": full_dates,
        "price_usd": np.linspace(30000.0, 50000.0, full_dates.len()),
    })
    features_df = pl.DataFrame({
        "date": full_dates,
        "price_usd": np.linspace(30000.0, 50000.0, full_dates.len()),
    })

    max_start_date = end_ts - prelude_module.WINDOW_OFFSET
    real_datetime_range = pl.datetime_range

    def _fake_datetime_range(start, end, interval="1d", eager=True):
        if start == start_ts and end == max_start_date:
            return pl.Series([start_ts])
        return real_datetime_range(start, end, interval=interval, eager=eager)

    monkeypatch.setattr("stacksats.prelude.pl.datetime_range", _fake_datetime_range)

    calls = {"count": 0}

    def _strategy(window: pl.DataFrame) -> pl.DataFrame:
        calls["count"] += 1
        return pl.DataFrame({
            "date": window["date"],
            "weight": [1.0 / window.height] * window.height,
        })

    result = compute_cycle_spd(
        btc_df,
        strategy_function=_strategy,
        features_df=features_df,
        start_date=start_ts.strftime("%Y-%m-%d"),
        end_date=end_ts.strftime("%Y-%m-%d"),
        validate_weights=True,
    )

    assert calls["count"] == 1
    assert result.height == 1


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
    dates = pl.datetime_range(
        dt.datetime(2024, 1, 1),
        dt.datetime(2024, 1, 4),
        interval="1d",
        eager=True,
    )
    features_df = pl.DataFrame({
        "date": dates,
        "price_vs_ma": [0.1, -0.1, 0.2, -0.2],
        "mvrv_zscore": [0.5, -0.5, 1.0, -1.0],
        "mvrv_gradient": [0.2, -0.2, 0.3, -0.3],
    })

    scores = compute_preference_scores(
        features_df,
        dt.datetime(2024, 1, 1),
        dt.datetime(2024, 1, 4),
    )
    assert scores.height == 4
    pref_col = "preference" if "preference" in scores.columns else scores.columns[-1]
    assert np.isfinite(scores[pref_col].to_numpy().astype(float)).all()


def test_compute_weights_from_target_profile_returns_empty_for_empty_range() -> None:
    result = compute_weights_from_target_profile(
        features_df=pl.DataFrame(),
        start_date=dt.datetime(2024, 1, 2),
        end_date=dt.datetime(2024, 1, 1),
        current_date=dt.datetime(2024, 1, 1),
        target_profile=pl.DataFrame(schema={"date": pl.Datetime("us"), "value": pl.Float64}),
        mode="preference",
    )
    assert result.is_empty()


def test_compute_weights_from_target_profile_absolute_uses_base_when_all_nonpositive() -> None:
    dates = pl.datetime_range(
        dt.datetime(2024, 1, 1),
        dt.datetime(2024, 1, 4),
        interval="1d",
        eager=True,
    )
    target_profile = pl.DataFrame({
        "date": dates,
        "value": [-5.0, 0.0, -1.0, float("nan")],
    })

    weights = compute_weights_from_target_profile(
        features_df=pl.DataFrame({"date": dates}),
        start_date=dates.min(),
        end_date=dates.max(),
        current_date=dates.max(),
        target_profile=target_profile,
        mode="absolute",
    )

    np.testing.assert_allclose(
        weights["weight"].to_numpy().astype(float),
        np.full(4, 1.0 / 4),
    )


def test_compute_weights_from_target_profile_rejects_unsupported_mode() -> None:
    dates = pl.datetime_range(
        dt.datetime(2024, 1, 1),
        dt.datetime(2024, 1, 3),
        interval="1d",
        eager=True,
    )
    with pytest.raises(ValueError, match="Unsupported target profile mode"):
        compute_weights_from_target_profile(
            features_df=pl.DataFrame({"date": dates}),
            start_date=dates.min(),
            end_date=dates.max(),
            current_date=dates.max(),
            target_profile=pl.DataFrame({"date": dates, "value": [0.0, 0.0, 0.0]}),
            mode="unknown",
        )


def test_compute_weights_from_proposals_returns_empty_for_empty_range() -> None:
    result = compute_weights_from_proposals(
        proposals=pl.DataFrame(schema={"date": pl.Datetime("us"), "value": pl.Float64}),
        start_date=dt.datetime(2024, 1, 2),
        end_date=dt.datetime(2024, 1, 1),
        n_past=0,
    )
    assert result.is_empty()


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
    dates = pl.datetime_range(
        dt.datetime(2024, 1, 1),
        dt.datetime(2024, 1, 1) + dt.timedelta(days=periods - 1),
        interval="1d",
        eager=True,
    )
    features_df = pl.DataFrame({
        "date": dates,
        "price_usd": np.linspace(100.0, 110.0, periods),
    })
    return strategy_context_from_features_df(
        features_df,
        dates.min(),
        dates.max(),
        dates.max(),
    )


def test_base_strategy_default_build_target_profile_returns_absolute_profile() -> None:
    strategy = _UniformProposalStrategy()
    ctx = _strategy_context(periods=4)
    features_df = strategy.transform_features(ctx)

    profile = strategy.build_target_profile(ctx, features_df, signals={})

    assert isinstance(profile, TargetProfile)
    assert profile.mode == "absolute"
    assert profile.values.height == 4
    assert np.isfinite(profile.values["value"].to_numpy().astype(float)).all()


def test_base_strategy_default_build_target_profile_handles_empty_features() -> None:
    strategy = _UniformProposalStrategy()
    ctx = _strategy_context(periods=3)
    empty_df = pl.DataFrame(schema={"date": pl.Datetime("us"), "price_usd": pl.Float64})
    profile = strategy.build_target_profile(ctx, empty_df, signals={})

    assert profile.mode == "absolute"
    assert profile.values.is_empty()


def test_base_strategy_default_build_target_profile_rejects_nonfinite_proposals() -> None:
    strategy = _NanProposalStrategy()
    ctx = _strategy_context(periods=3)
    features_df = strategy.transform_features(ctx)

    with pytest.raises(ValueError, match="finite numeric value"):
        strategy.build_target_profile(ctx, features_df, signals={})


def test_base_strategy_validate_weights_allows_empty_weights() -> None:
    strategy = _UniformProposalStrategy()
    strategy.validate_weights(
        pl.DataFrame(schema={"date": pl.Datetime("us"), "weight": pl.Float64}),
        _strategy_context(),
    )
