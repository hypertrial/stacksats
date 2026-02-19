from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stacksats.strategy_time_series import StrategySeriesMetadata, StrategyTimeSeries, StrategyTimeSeriesBatch


def _metadata(
    *,
    strategy_id: str = "s1",
    strategy_version: str = "1.0.0",
    run_id: str = "run-1",
    config_hash: str = "cfg-1",
    schema_version: str = "1.0.0",
    window_start: pd.Timestamp | None = pd.Timestamp("2024-01-01"),
    window_end: pd.Timestamp | None = pd.Timestamp("2024-01-02"),
) -> StrategySeriesMetadata:
    return StrategySeriesMetadata(
        strategy_id=strategy_id,
        strategy_version=strategy_version,
        run_id=run_id,
        config_hash=config_hash,
        schema_version=schema_version,
        window_start=window_start,
        window_end=window_end,
    )


def _valid_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=2, freq="D"),
            "weight": [0.4, 0.6],
            "price_usd": [40000.0, 41000.0],
        }
    )


def _window(md: StrategySeriesMetadata | None = None) -> StrategyTimeSeries:
    return StrategyTimeSeries(metadata=md or _metadata(), data=_valid_data())


def test_strategy_time_series_requires_dataframe_data() -> None:
    with pytest.raises(TypeError, match="must be a pandas DataFrame"):
        StrategyTimeSeries(metadata=_metadata(), data=[1, 2])  # type: ignore[arg-type]


def test_strategy_time_series_rejects_missing_required_columns() -> None:
    missing_price = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=2, freq="D"),
            "weight": [0.4, 0.6],
        }
    )
    with pytest.raises(ValueError, match="missing required columns"):
        StrategyTimeSeries(metadata=_metadata(), data=missing_price)


def test_strategy_time_series_rejects_invalid_dates() -> None:
    bad_date = pd.DataFrame(
        {
            "date": ["2024-01-01", "not-a-date"],
            "weight": [0.4, 0.6],
            "price_usd": [40000.0, 41000.0],
        }
    )
    with pytest.raises(ValueError, match="must contain valid datetimes"):
        StrategyTimeSeries(metadata=_metadata(), data=bad_date)


def test_strategy_time_series_rejects_duplicate_dates() -> None:
    duplicated = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01"],
            "weight": [0.4, 0.6],
            "price_usd": [40000.0, 41000.0],
        }
    )
    with pytest.raises(ValueError, match="must not contain duplicates"):
        StrategyTimeSeries(metadata=_metadata(), data=duplicated)


def test_strategy_time_series_rejects_unsorted_dates() -> None:
    series = _window()
    series.data.loc[:, "date"] = [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-01")]
    with pytest.raises(ValueError, match="must be sorted ascending"):
        series.validate()


def test_strategy_time_series_rejects_window_start_mismatch() -> None:
    with pytest.raises(ValueError, match="start date does not match metadata.window_start"):
        StrategyTimeSeries(
            metadata=_metadata(window_start=pd.Timestamp("2024-01-02"), window_end=pd.Timestamp("2024-01-02")),
            data=_valid_data(),
        )


def test_strategy_time_series_rejects_window_end_mismatch() -> None:
    with pytest.raises(ValueError, match="end date does not match metadata.window_end"):
        StrategyTimeSeries(
            metadata=_metadata(window_start=pd.Timestamp("2024-01-01"), window_end=pd.Timestamp("2024-01-03")),
            data=_valid_data(),
        )


def test_strategy_time_series_rejects_nonfinite_weights() -> None:
    nonfinite_weights = _valid_data()
    nonfinite_weights["weight"] = [0.4, np.inf]
    with pytest.raises(ValueError, match="must contain finite numeric values"):
        StrategyTimeSeries(metadata=_metadata(), data=nonfinite_weights)


def test_strategy_time_series_rejects_negative_weights() -> None:
    negative_weights = _valid_data()
    negative_weights["weight"] = [-0.1, 1.1]
    with pytest.raises(ValueError, match="must not contain negative values"):
        StrategyTimeSeries(metadata=_metadata(), data=negative_weights)


def test_strategy_time_series_rejects_nonnumeric_price_usd() -> None:
    series = _window()
    series.data["price_usd"] = series.data["price_usd"].astype(object)
    series.data["price_usd"] = [40000.0, "bad"]
    with pytest.raises(ValueError, match="price_usd' must be numeric"):
        series.validate()


def test_strategy_time_series_rejects_nonfinite_price_usd() -> None:
    nonfinite_price = _valid_data()
    nonfinite_price["price_usd"] = [40000.0, np.inf]
    with pytest.raises(ValueError, match="price_usd' must be finite"):
        StrategyTimeSeries(metadata=_metadata(), data=nonfinite_price)


def test_strategy_time_series_rejects_nonboolean_locked_column() -> None:
    bad_locked = _valid_data()
    bad_locked["locked"] = [True, "bad"]
    with pytest.raises(ValueError, match="locked' must contain only boolean values"):
        StrategyTimeSeries(metadata=_metadata(), data=bad_locked)


def test_strategy_time_series_rejects_nonnumeric_day_index() -> None:
    bad_day_index = _valid_data()
    bad_day_index["day_index"] = [0, "bad"]
    with pytest.raises(ValueError, match="day_index' must contain integer values"):
        StrategyTimeSeries(metadata=_metadata(), data=bad_day_index)


def test_strategy_time_series_rejects_negative_day_index() -> None:
    negative_day_index = _valid_data()
    negative_day_index["day_index"] = [-1, 0]
    with pytest.raises(ValueError, match="day_index' must be >= 0"):
        StrategyTimeSeries(metadata=_metadata(), data=negative_day_index)


def test_strategy_time_series_rejects_noncontiguous_day_index() -> None:
    noncontiguous_day_index = _valid_data()
    noncontiguous_day_index["day_index"] = [0, 2]
    with pytest.raises(ValueError, match="must be contiguous starting at 0"):
        StrategyTimeSeries(metadata=_metadata(), data=noncontiguous_day_index)


def test_strategy_time_series_validates_optional_datetime_columns() -> None:
    with_time = _valid_data()
    with_time["time"] = pd.date_range("2024-01-01", periods=2, freq="D")
    series = StrategyTimeSeries(metadata=_metadata(), data=with_time)
    assert "time" in series.data.columns


def test_strategy_time_series_rejects_invalid_optional_numeric_column() -> None:
    bad_optional_numeric = _valid_data()
    bad_optional_numeric["SplyCur"] = [100.0, "bad"]
    with pytest.raises(ValueError, match="SplyCur' must be numeric"):
        StrategyTimeSeries(metadata=_metadata(), data=bad_optional_numeric)


def test_strategy_time_series_rejects_nonfinite_optional_numeric_column() -> None:
    nonfinite_optional_numeric = _valid_data()
    nonfinite_optional_numeric["SplyCur"] = [100.0, np.inf]
    with pytest.raises(ValueError, match="SplyCur' must be finite"):
        StrategyTimeSeries(metadata=_metadata(), data=nonfinite_optional_numeric)


def test_strategy_time_series_rejects_invalid_optional_datetime_column() -> None:
    bad_optional_datetime = _valid_data()
    bad_optional_datetime["time"] = ["2024-01-01", "not-a-date"]
    with pytest.raises(ValueError, match="time' must be datetime"):
        StrategyTimeSeries(metadata=_metadata(), data=bad_optional_datetime)


def test_strategy_time_series_batch_requires_windows() -> None:
    with pytest.raises(ValueError, match="windows must not be empty"):
        StrategyTimeSeriesBatch(
            strategy_id="s1",
            strategy_version="1.0.0",
            run_id="run-1",
            config_hash="cfg-1",
            windows=(),
        )


def test_batch_from_flat_dataframe_requires_dataframe() -> None:
    with pytest.raises(TypeError, match="must be a pandas DataFrame"):
        StrategyTimeSeriesBatch.from_flat_dataframe(  # type: ignore[arg-type]
            [{"a": 1}],
            strategy_id="s1",
            strategy_version="1.0.0",
            run_id="run-1",
            config_hash="cfg-1",
        )


def test_batch_from_flat_dataframe_requires_columns() -> None:
    with pytest.raises(ValueError, match="Flat dataframe missing required columns"):
        StrategyTimeSeriesBatch.from_flat_dataframe(
            pd.DataFrame(
                {
                    "start_date": ["2024-01-01"],
                    "end_date": ["2024-01-02"],
                    "date": ["2024-01-01"],
                    "weight": [1.0],
                }
            ),
            strategy_id="s1",
            strategy_version="1.0.0",
            run_id="run-1",
            config_hash="cfg-1",
        )


def test_batch_from_flat_dataframe_requires_valid_datetimes() -> None:
    with pytest.raises(ValueError, match="must be valid datetimes"):
        StrategyTimeSeriesBatch.from_flat_dataframe(
            pd.DataFrame(
                {
                    "start_date": ["not-a-date"],
                    "end_date": ["2024-01-02"],
                    "date": ["2024-01-01"],
                    "weight": [1.0],
                    "price_usd": [40000.0],
                }
            ),
            strategy_id="s1",
            strategy_version="1.0.0",
            run_id="run-1",
            config_hash="cfg-1",
        )


def test_batch_validate_rejects_strategy_id_mismatch() -> None:
    with pytest.raises(ValueError, match="strategy_id does not match"):
        StrategyTimeSeriesBatch(
            strategy_id="other",
            strategy_version="1.0.0",
            run_id="run-1",
            config_hash="cfg-1",
            windows=(_window(_metadata(strategy_id="s1")),),
        )


def test_batch_validate_rejects_strategy_version_mismatch() -> None:
    with pytest.raises(ValueError, match="strategy_version does not match"):
        StrategyTimeSeriesBatch(
            strategy_id="s1",
            strategy_version="2.0.0",
            run_id="run-1",
            config_hash="cfg-1",
            windows=(_window(_metadata(strategy_version="1.0.0")),),
        )


def test_batch_validate_rejects_run_id_mismatch() -> None:
    with pytest.raises(ValueError, match="run_id does not match"):
        StrategyTimeSeriesBatch(
            strategy_id="s1",
            strategy_version="1.0.0",
            run_id="run-2",
            config_hash="cfg-1",
            windows=(_window(_metadata(run_id="run-1")),),
        )


def test_batch_validate_rejects_config_hash_mismatch() -> None:
    with pytest.raises(ValueError, match="config_hash does not match"):
        StrategyTimeSeriesBatch(
            strategy_id="s1",
            strategy_version="1.0.0",
            run_id="run-1",
            config_hash="cfg-2",
            windows=(_window(_metadata(config_hash="cfg-1")),),
        )


def test_batch_validate_rejects_schema_version_mismatch() -> None:
    with pytest.raises(ValueError, match="schema_version does not match"):
        StrategyTimeSeriesBatch(
            strategy_id="s1",
            strategy_version="1.0.0",
            run_id="run-1",
            config_hash="cfg-1",
            schema_version="2.0.0",
            windows=(_window(_metadata(schema_version="1.0.0")),),
        )


def test_batch_validate_requires_window_bounds() -> None:
    with pytest.raises(ValueError, match="must define metadata.window_start and metadata.window_end"):
        StrategyTimeSeriesBatch(
            strategy_id="s1",
            strategy_version="1.0.0",
            run_id="run-1",
            config_hash="cfg-1",
            windows=(_window(_metadata(window_start=None, window_end=None)),),
        )


def test_batch_iter_windows_returns_iterator() -> None:
    batch = StrategyTimeSeriesBatch(
        strategy_id="s1",
        strategy_version="1.0.0",
        run_id="run-1",
        config_hash="cfg-1",
        windows=(_window(),),
    )
    windows = list(batch.iter_windows())
    assert len(windows) == 1


def test_batch_for_window_raises_key_error_when_missing() -> None:
    batch = StrategyTimeSeriesBatch(
        strategy_id="s1",
        strategy_version="1.0.0",
        run_id="run-1",
        config_hash="cfg-1",
        windows=(_window(),),
    )
    with pytest.raises(KeyError, match="Window not found"):
        batch.for_window("2030-01-01", "2030-01-02")


def test_strategy_time_series_outlier_report_rejects_unknown_method() -> None:
    series = _window()
    with pytest.raises(ValueError, match="method must be one of"):
        series.outlier_report(method="bogus")


def test_strategy_time_series_outlier_report_rejects_unknown_columns() -> None:
    series = _window()
    with pytest.raises(ValueError, match="Unknown columns"):
        series.outlier_report(columns=["unknown_col"])


def test_strategy_time_series_outlier_report_returns_empty_with_constant_numeric_columns() -> None:
    data = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3, freq="D"),
            "weight": [1 / 3, 1 / 3, 1 / 3],
            "price_usd": [100.0, 100.0, 100.0],
            "SplyCur": [10.0, 10.0, 10.0],
        }
    )
    series = StrategyTimeSeries(
        metadata=_metadata(window_end=pd.Timestamp("2024-01-03")),
        data=data,
    )

    report = series.outlier_report(method="zscore")

    assert report.empty
    assert list(report.columns) == ["date", "column", "value", "score", "method", "threshold"]


def test_strategy_time_series_returns_diagnostics_handles_short_or_invalid_prices() -> None:
    data = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=2, freq="D"),
            "weight": [0.4, 0.6],
            "price_usd": [40000.0, np.nan],
        }
    )
    series = StrategyTimeSeries(metadata=_metadata(), data=data)

    diagnostics = series.returns_diagnostics()

    assert diagnostics["price_observations"] == 1
    assert diagnostics["return_observations"] == 0
    assert diagnostics["cumulative_return"] is None
    assert diagnostics["std_simple_return"] is None
    assert diagnostics["annualized_volatility"] is None


def test_strategy_time_series_outlier_report_rejects_non_positive_threshold() -> None:
    series = _window()
    with pytest.raises(ValueError, match="threshold must be > 0"):
        series.outlier_report(threshold=0.0)


def test_strategy_time_series_outlier_report_iqr_uses_default_threshold_and_case_insensitive_method() -> None:
    data = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "weight": [0.2, 0.2, 0.2, 0.2, 0.2],
            "price_usd": [100.0, 101.0, 102.0, 103.0, 1000.0],
        }
    )
    series = StrategyTimeSeries(
        metadata=_metadata(window_end=pd.Timestamp("2024-01-05")),
        data=data,
    )

    report = series.outlier_report(method="IQR")

    assert len(report) == 1
    assert report.iloc[0]["method"] == "iqr"
    assert np.isclose(report.iloc[0]["threshold"], 1.5)


def test_strategy_time_series_weight_diagnostics_handles_negative_top_k() -> None:
    series = _window()

    diagnostics = series.weight_diagnostics(top_k=-5)

    assert diagnostics["sample_size"] == 2
    assert diagnostics["top_weights"] == []


def test_strategy_time_series_returns_diagnostics_reports_drawdown_metrics() -> None:
    data = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3, freq="D"),
            "weight": [0.2, 0.3, 0.5],
            "price_usd": [100.0, 90.0, 95.0],
        }
    )
    series = StrategyTimeSeries(
        metadata=_metadata(window_end=pd.Timestamp("2024-01-03")),
        data=data,
    )

    diagnostics = series.returns_diagnostics()

    assert np.isclose(diagnostics["max_drawdown"], -0.1)
    assert diagnostics["max_drawdown_date"] == "2024-01-02T00:00:00"
    assert diagnostics["best_day_date"] == "2024-01-03T00:00:00"
    assert diagnostics["worst_day_date"] == "2024-01-02T00:00:00"


def test_strategy_time_series_rolling_statistics_rejects_invalid_windows() -> None:
    series = _window()
    with pytest.raises(ValueError, match="windows must contain only positive integers"):
        series.rolling_statistics(windows=(0, 7))


def test_strategy_time_series_rolling_statistics_rejects_unknown_price_column() -> None:
    series = _window()
    with pytest.raises(ValueError, match="Unknown price column"):
        series.rolling_statistics(price_col="unknown_price")


def test_strategy_time_series_autocorrelation_rejects_invalid_lags() -> None:
    series = _window()
    with pytest.raises(ValueError, match="lags must contain only positive integers"):
        series.autocorrelation(lags=(1, 0))


def test_strategy_time_series_autocorrelation_rejects_unknown_series() -> None:
    series = _window()
    with pytest.raises(ValueError, match="series must be one of"):
        series.autocorrelation(series="unknown")


def test_strategy_time_series_autocorrelation_returns_none_for_large_lag() -> None:
    series = _window()
    acf = series.autocorrelation(lags=(10,), series="returns")
    assert acf["autocorrelation"]["10"] is None


def test_strategy_time_series_drawdown_table_rejects_invalid_top_n() -> None:
    series = _window()
    with pytest.raises(ValueError, match="top_n must be > 0"):
        series.drawdown_table(top_n=0)


def test_strategy_time_series_drawdown_table_returns_empty_when_no_drawdowns() -> None:
    data = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3, freq="D"),
            "weight": [0.2, 0.3, 0.5],
            "price_usd": [100.0, 101.0, 102.0],
        }
    )
    series = StrategyTimeSeries(
        metadata=_metadata(window_end=pd.Timestamp("2024-01-03")),
        data=data,
    )

    drawdowns = series.drawdown_table()

    assert drawdowns.empty
    assert list(drawdowns.columns) == [
        "peak_date",
        "trough_date",
        "recovery_date",
        "max_drawdown",
        "days_to_trough",
        "days_to_recovery",
        "duration_days",
        "recovered",
    ]


def test_strategy_time_series_seasonality_profile_rejects_invalid_frequency() -> None:
    series = _window()
    with pytest.raises(ValueError, match="freq must be one of"):
        series.seasonality_profile(freq="quarter")


def test_strategy_time_series_seasonality_profile_month_returns_12_rows() -> None:
    data = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-02-01"],
            "weight": [0.4, 0.6],
            "price_usd": [100.0, 110.0],
        }
    )
    series = StrategyTimeSeries(
        metadata=_metadata(window_end=pd.Timestamp("2024-02-01")),
        data=data,
    )

    profile = series.seasonality_profile(freq="month", series="price")

    assert len(profile) == 12
    jan = profile.loc[profile["period_label"] == "Jan"].iloc[0]
    feb = profile.loc[profile["period_label"] == "Feb"].iloc[0]
    mar = profile.loc[profile["period_label"] == "Mar"].iloc[0]
    assert int(jan["count"]) == 1
    assert int(feb["count"]) == 1
    assert int(mar["count"]) == 0
