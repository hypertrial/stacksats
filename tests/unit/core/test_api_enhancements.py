"""Tests for enhanced public API functionality."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from stacksats.api import BacktestResult, ValidationResult
from stacksats.strategy_types import BaseStrategy, StrategyContext, TargetProfile, ValidationConfig
from stacksats.strategies.examples import (
    MomentumStrategy,
    SimpleZScoreStrategy,
    UniformStrategy,
)


def _sample_btc_df() -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", periods=520, freq="D")
    price = np.linspace(20000.0, 45000.0, len(dates))
    mvrv = np.linspace(0.8, 2.2, len(dates))
    return pd.DataFrame(
        {
            "PriceUSD_coinmetrics": price,
            "CapMVRVCur": mvrv,
        },
        index=dates,
    )


def _sample_spd_df() -> pd.DataFrame:
    windows = [
        "2022-01-01 → 2023-01-01",
        "2022-02-01 → 2023-02-01",
        "2022-03-01 → 2023-03-01",
    ]
    return pd.DataFrame(
        {
            "min_sats_per_dollar": [1000.0, 1200.0, 900.0],
            "max_sats_per_dollar": [5000.0, 5200.0, 4800.0],
            "uniform_sats_per_dollar": [2500.0, 2600.0, 2300.0],
            "dynamic_sats_per_dollar": [2800.0, 2900.0, 2400.0],
            "uniform_percentile": [37.5, 35.0, 36.8],
            "dynamic_percentile": [45.0, 42.0, 39.5],
            "excess_percentile": [7.5, 7.0, 2.7],
        },
        index=windows,
    )


def test_backtest_result_summary_dataframe_and_json(tmp_path: Path):
    """BacktestResult helper methods should return stable structured outputs."""
    result = BacktestResult(
        spd_table=_sample_spd_df(),
        exp_decay_percentile=44.2,
        win_rate=66.6,
        score=55.4,
    )

    summary = result.summary()
    assert "Score: 55.40%" in summary
    assert "Win Rate: 66.60%" in summary

    as_df = result.to_dataframe()
    assert as_df.equals(result.spd_table)
    assert as_df is not result.spd_table

    output_path = tmp_path / "result.json"
    payload = result.to_json(output_path)
    assert output_path.exists()
    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted == payload
    assert payload["summary_metrics"]["score"] == 55.4
    assert len(payload["window_level_data"]) == len(result.spd_table)


def test_backtest_result_plot_delegates_to_backtest_module(mocker):
    """BacktestResult.plot should call all plot/export helpers exactly once."""
    result = BacktestResult(
        spd_table=_sample_spd_df(),
        exp_decay_percentile=44.2,
        win_rate=66.6,
        score=55.4,
    )

    mocked_calls = [
        mocker.patch("stacksats.backtest.create_performance_comparison_chart"),
        mocker.patch("stacksats.backtest.create_excess_percentile_distribution"),
        mocker.patch("stacksats.backtest.create_win_loss_comparison"),
        mocker.patch("stacksats.backtest.create_cumulative_performance"),
        mocker.patch("stacksats.backtest.create_performance_metrics_summary"),
        mocker.patch("stacksats.backtest.export_metrics_json"),
    ]

    paths = result.plot(output_dir="my-output")

    for mocked in mocked_calls:
        mocked.assert_called_once()
    assert paths["metrics_json"].endswith("my-output/metrics.json")
    assert paths["performance_comparison"].endswith("my-output/performance_comparison.svg")


def test_validation_result_summary_format():
    """ValidationResult summary should include primary gate statuses."""
    result = ValidationResult(
        passed=True,
        forward_leakage_ok=True,
        weight_constraints_ok=True,
        win_rate=72.3,
        win_rate_ok=True,
        messages=["All checks passed."],
    )

    summary = result.summary()
    assert "Validation PASSED" in summary
    assert "Forward Leakage: True" in summary
    assert "Weight Constraints: True" in summary
    assert "Win Rate: 72.30%" in summary


def test_validate_strategy_passes_with_uniform_strategy():
    """Uniform example strategy should satisfy validation when win-rate floor is relaxed."""
    btc_df = _sample_btc_df()
    result = UniformStrategy().validate(
        ValidationConfig(
            start_date="2022-01-01",
            end_date="2023-05-01",
            min_win_rate=0.0,
        ),
        btc_df=btc_df,
    )

    assert result.forward_leakage_ok is True
    assert result.weight_constraints_ok is True
    assert bool(result.win_rate_ok) is True
    assert bool(result.passed) is True


def test_validate_strategy_fails_weight_constraints_for_bad_strategy():
    """Invalid strategy should fail at backtest assertion on weight sums."""
    btc_df = _sample_btc_df()

    class BadWeightsStrategy(BaseStrategy):
        strategy_id = "bad-weights"
        version = "1.0.0"

        def build_target_profile(
            self,
            ctx: StrategyContext,
            features_df: pd.DataFrame,
            signals: dict[str, pd.Series],
        ) -> TargetProfile:
            del ctx, signals
            return TargetProfile(
                values=pd.Series(np.nan, index=features_df.index),
                mode="preference",
            )

    import pytest

    with pytest.raises(ValueError, match="target profile must contain finite numeric values"):
        BadWeightsStrategy().validate(
            ValidationConfig(
                start_date="2022-01-01",
                end_date="2023-05-01",
                min_win_rate=0.0,
            ),
            btc_df=btc_df,
        )


def test_validate_strategy_fails_forward_leakage_for_peeking_strategy():
    """Validation should detect a strategy that peeks beyond window end."""
    btc_df = _sample_btc_df()

    class LeakyStrategy(BaseStrategy):
        strategy_id = "leaky"
        version = "1.0.0"

        def build_target_profile(
            self,
            ctx: StrategyContext,
            features_df: pd.DataFrame,
            signals: dict[str, pd.Series],
        ) -> pd.Series:
            del signals
            idx = features_df.index
            preference = pd.Series(0.0, index=idx)
            lookahead_date = pd.Timestamp(ctx.end_date) + pd.Timedelta(days=1)
            future_signal = 0.0
            if lookahead_date in ctx.features_df.index:
                future_signal = float(ctx.features_df.loc[lookahead_date, "price_vs_ma"])
                if np.isnan(future_signal):
                    future_signal = 0.0
            preference.iloc[-1] = future_signal
            return preference

    result = LeakyStrategy().validate(
        ValidationConfig(
            start_date="2022-01-01",
            end_date="2023-05-01",
            min_win_rate=0.0,
        ),
        btc_df=btc_df,
    )
    assert result.forward_leakage_ok is False
    assert result.passed is False
    assert any("Forward leakage detected" in msg for msg in result.messages)


def test_example_strategies_return_valid_weight_vectors():
    """Example strategies should produce non-negative, normalized weights."""
    btc_df = _sample_btc_df()
    from stacksats.model_development import precompute_features

    features_df = precompute_features(btc_df)
    start_date = pd.Timestamp("2022-01-01")
    end_date = pd.Timestamp("2022-12-31")

    for strategy in (UniformStrategy(), SimpleZScoreStrategy(), MomentumStrategy()):
        weights = strategy.compute_weights(
            StrategyContext(
                features_df=features_df,
                start_date=start_date,
                end_date=end_date,
                current_date=end_date,
            )
        )
        assert not weights.empty
        assert bool((weights >= 0).all())
        assert np.isclose(float(weights.sum()), 1.0, atol=1e-8)
