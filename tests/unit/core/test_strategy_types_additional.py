from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stacksats.framework_contract import ALLOCATION_SPAN_DAYS, MAX_DAILY_WEIGHT, MIN_DAILY_WEIGHT
from stacksats.strategy_types import BacktestConfig, BaseStrategy, DayState, StrategyContext


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


def test_validate_weights_rejects_values_below_contract_min() -> None:
    strategy = _SimpleProposeStrategy()
    ctx = _context()
    idx = pd.date_range("2024-01-01", periods=ALLOCATION_SPAN_DAYS, freq="D")

    base = 1.0 / ALLOCATION_SPAN_DAYS
    weights = np.full(ALLOCATION_SPAN_DAYS, base, dtype=float)
    weights[0] = MIN_DAILY_WEIGHT / 10.0
    deficit = base - weights[0]
    weights[1:] += deficit / (ALLOCATION_SPAN_DAYS - 1)

    with pytest.raises(ValueError, match="must be >="):
        strategy.validate_weights(pd.Series(weights, index=idx), ctx)


def test_validate_weights_rejects_values_above_contract_max() -> None:
    strategy = _SimpleProposeStrategy()
    ctx = _context()
    idx = pd.date_range("2024-01-01", periods=ALLOCATION_SPAN_DAYS, freq="D")

    base = 1.0 / ALLOCATION_SPAN_DAYS
    weights = np.full(ALLOCATION_SPAN_DAYS, base, dtype=float)
    weights[0] = MAX_DAILY_WEIGHT + 1e-3
    excess = weights[0] - base
    weights[1:] -= excess / (ALLOCATION_SPAN_DAYS - 1)

    with pytest.raises(ValueError, match="must be <="):
        strategy.validate_weights(pd.Series(weights, index=idx), ctx)


def test_default_config_methods_include_strategy_metadata() -> None:
    strategy = _SimpleProposeStrategy()

    assert strategy.default_backtest_config().strategy_label == strategy.strategy_id
    assert strategy.default_validation_config().min_win_rate == 50.0
    export_config = strategy.default_export_config()
    start = pd.to_datetime(export_config.range_start)
    end = pd.to_datetime(export_config.range_end)
    assert end >= start
    assert (end - start).days in {364, 365}


def test_strategy_wrapper_methods_delegate_to_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    strategy = _SimpleProposeStrategy()
    backtest_result = object()
    validation_result = object()
    export_result = object()

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
    assert strategy.export_weights() is export_result
    assert strategy.export() is export_result


def test_strategy_contract_helper_methods_reflect_hook_status() -> None:
    strategy = _SimpleProposeStrategy()
    has_propose_hook, has_profile_hook = strategy.hook_status()
    assert has_propose_hook is True
    assert has_profile_hook is False

    validated_propose, validated_profile = strategy.validate_contract()
    assert validated_propose is True
    assert validated_profile is False


def test_backtest_and_save_writes_standard_artifacts(tmp_path) -> None:
    strategy = _SimpleProposeStrategy()

    class FakeBacktestResult:
        strategy_id = "fake-strategy"
        strategy_version = "9.9.9"
        run_id = "run-123"

        def __init__(self):
            self.plot_calls = 0
            self.json_calls = 0

        def plot(self, output_dir="output"):
            self.plot_calls += 1
            return {"output_dir": output_dir}

        def to_json(self, path=None):
            self.json_calls += 1
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}", encoding="utf-8")
            return {}

    fake_result = FakeBacktestResult()

    def _fake_backtest(*args, **kwargs):
        del args, kwargs
        return fake_result

    strategy.backtest = _fake_backtest  # type: ignore[method-assign]

    result, output_dir = strategy.backtest_and_save(
        config=BacktestConfig(),
        output_dir=str(tmp_path),
    )
    expected_root = tmp_path / "fake-strategy" / "9.9.9" / "run-123"

    assert result is fake_result
    assert output_dir == str(expected_root)
    assert expected_root.exists()
    assert (expected_root / "backtest_result.json").exists()
    assert fake_result.plot_calls == 1
    assert fake_result.json_calls == 1


def test_run_executes_lifecycle_with_optional_export_and_save() -> None:
    strategy = _SimpleProposeStrategy()
    events = []
    validation_result = object()
    backtest_result = object()
    export_result = object()

    def _fake_validate(*args, **kwargs):
        del args, kwargs
        events.append("validate")
        return validation_result

    def _fake_backtest_and_save(*args, **kwargs):
        del args, kwargs
        events.append("backtest_and_save")
        return backtest_result, "saved-output"

    def _fake_export(*args, **kwargs):
        del args, kwargs
        events.append("export")
        return export_result

    strategy.validate = _fake_validate  # type: ignore[method-assign]
    strategy.backtest_and_save = _fake_backtest_and_save  # type: ignore[method-assign]
    strategy.export = _fake_export  # type: ignore[method-assign]

    run_result = strategy.run(
        include_export=True,
        save_backtest_artifacts=True,
    )

    assert events == ["validate", "backtest_and_save", "export"]
    assert run_result.validation is validation_result
    assert run_result.backtest is backtest_result
    assert run_result.export_batch is export_result
    assert run_result.output_dir == "saved-output"


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


def test_base_strategy_default_propose_weight_raises_not_implemented() -> None:
    class _ProfileOnlyStrategy(BaseStrategy):
        strategy_id = "profile-only"

        def build_target_profile(self, ctx, features_df, signals):
            del ctx, features_df, signals
            return pd.Series(dtype=float)

    state = DayState(
        current_date=pd.Timestamp("2024-01-01"),
        features=pd.Series(dtype=float),
        remaining_budget=1.0,
        day_index=0,
        total_days=1,
        uniform_weight=1.0,
    )

    with pytest.raises(NotImplementedError, match="Override propose_weight"):
        _ProfileOnlyStrategy().propose_weight(state)
