from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stacksats.strategy_types import BaseStrategy, StrategyContext


def _context(*, start: str = "2024-01-01", periods: int = 3) -> StrategyContext:
    idx = pd.date_range(start, periods=periods, freq="D")
    features_df = pd.DataFrame(
        {
            "PriceUSD_coinmetrics": np.linspace(100.0, 102.0, len(idx)),
            "CapMVRVCur": np.linspace(1.0, 1.2, len(idx)),
        },
        index=idx,
    )
    return StrategyContext(
        features_df=features_df,
        start_date=idx.min(),
        end_date=idx.max(),
        current_date=idx.max(),
    )


class _SimpleProposeStrategy(BaseStrategy):
    strategy_id = "simple-propose"

    def propose_weight(self, state):
        return state.uniform_weight


def test_validate_series_requires_pandas_series() -> None:
    strategy = _SimpleProposeStrategy()
    idx = pd.date_range("2024-01-01", periods=2, freq="D")

    with pytest.raises(TypeError, match="must be a pandas Series"):
        strategy._validate_series([1.0, 2.0], name="series", expected_index=idx)


def test_validate_series_rejects_duplicate_index() -> None:
    strategy = _SimpleProposeStrategy()
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    values = pd.Series([1.0, 2.0], index=[idx[0], idx[0]])

    with pytest.raises(ValueError, match="must not contain duplicates"):
        strategy._validate_series(values, name="series", expected_index=idx)


def test_validate_series_rejects_unsorted_index() -> None:
    strategy = _SimpleProposeStrategy()
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    values = pd.Series([1.0, 2.0], index=[idx[1], idx[0]])

    with pytest.raises(ValueError, match="must be sorted ascending"):
        strategy._validate_series(values, name="series", expected_index=idx)


def test_validate_series_rejects_mismatched_index() -> None:
    strategy = _SimpleProposeStrategy()
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    values = pd.Series([1.0, 2.0], index=pd.date_range("2024-02-01", periods=2, freq="D"))

    with pytest.raises(ValueError, match="must exactly match strategy window index"):
        strategy._validate_series(values, name="series", expected_index=idx)


def test_validate_series_rejects_non_finite_values() -> None:
    strategy = _SimpleProposeStrategy()
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    values = pd.Series([1.0, np.nan], index=idx)

    with pytest.raises(ValueError, match="must contain finite numeric values"):
        strategy._validate_series(values, name="series", expected_index=idx)


def test_compute_weights_rejects_non_dataframe_transform_features() -> None:
    class BadTransformStrategy(_SimpleProposeStrategy):
        def transform_features(self, ctx):
            del ctx
            return [1, 2, 3]

    with pytest.raises(TypeError, match="transform_features must return"):
        BadTransformStrategy().compute_weights(_context())


def test_compute_weights_rejects_non_dict_signals() -> None:
    class BadSignalsStrategy(_SimpleProposeStrategy):
        def build_signals(self, ctx, features_df):
            del ctx, features_df
            return []

    with pytest.raises(TypeError, match="build_signals must return"):
        BadSignalsStrategy().compute_weights(_context())


def test_compute_weights_rejects_invalid_signal_series() -> None:
    class BadSignalSeriesStrategy(_SimpleProposeStrategy):
        def build_signals(self, ctx, features_df):
            del ctx
            idx = features_df.index
            duplicated = pd.Series([1.0, 2.0], index=[idx[0], idx[0]])
            return {"bad": duplicated}

    with pytest.raises(ValueError, match="signal 'bad' index must not contain duplicates"):
        BadSignalSeriesStrategy().compute_weights(_context())


def test_compute_weights_rejects_invalid_target_profile_type() -> None:
    class BadProfileTypeStrategy(BaseStrategy):
        strategy_id = "bad-profile-type"

        def build_target_profile(self, ctx, features_df, signals):
            del ctx, features_df, signals
            return 123.0

    with pytest.raises(TypeError, match="target profile must be a pandas Series"):
        BadProfileTypeStrategy().compute_weights(_context())


def test_validate_weights_rejects_negative_values() -> None:
    strategy = _SimpleProposeStrategy()
    ctx = _context()
    weights = pd.Series([-0.1, 1.1], index=pd.date_range("2024-01-01", periods=2, freq="D"))

    with pytest.raises(ValueError, match="negative values"):
        strategy.validate_weights(weights, ctx)


def test_validate_weights_rejects_sum_mismatch() -> None:
    strategy = _SimpleProposeStrategy()
    ctx = _context()
    weights = pd.Series([0.4, 0.4], index=pd.date_range("2024-01-01", periods=2, freq="D"))

    with pytest.raises(ValueError, match="must sum to 1.0"):
        strategy.validate_weights(weights, ctx)


def test_default_config_methods_include_strategy_metadata() -> None:
    strategy = _SimpleProposeStrategy()

    assert strategy.default_backtest_config().strategy_label == strategy.strategy_id
    assert strategy.default_validation_config().min_win_rate == 50.0
    assert strategy.default_export_config().range_start == "2025-12-01"


def test_strategy_wrapper_methods_delegate_to_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    strategy = _SimpleProposeStrategy()
    backtest_result = object()
    validation_result = object()
    export_result = pd.DataFrame({"id": [1]})

    class FakeRunner:
        def backtest(self, *args, **kwargs):
            return backtest_result

        def validate(self, *args, **kwargs):
            return validation_result

        def export(self, *args, **kwargs):
            return export_result

    monkeypatch.setattr("stacksats.runner.StrategyRunner", lambda: FakeRunner())

    assert strategy.backtest() is backtest_result
    assert strategy.validate() is validation_result
    assert strategy.export_weights().equals(export_result)


def test_compute_weights_returns_empty_when_context_range_is_empty() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    ctx = StrategyContext(
        features_df=pd.DataFrame({"PriceUSD_coinmetrics": [100, 101, 102]}, index=idx),
        start_date=pd.Timestamp("2024-01-03"),
        end_date=pd.Timestamp("2024-01-01"),
        current_date=pd.Timestamp("2024-01-01"),
    )

    result = _SimpleProposeStrategy().compute_weights(ctx)
    assert result.empty
