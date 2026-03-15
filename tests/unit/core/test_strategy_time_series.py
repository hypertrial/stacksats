from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from stacksats.prelude import date_range_list
from stacksats.strategy_time_series import (
    StrategySeriesMetadata,
    StrategyTimeSeries,
    StrategyTimeSeriesBatch,
)


def _metadata() -> StrategySeriesMetadata:
    return StrategySeriesMetadata(
        strategy_id="test-strategy",
        strategy_version="1.2.3",
        run_id="run-1",
        config_hash="abc123",
        window_start=dt.datetime(2024, 1, 1),
        window_end=dt.datetime(2024, 1, 3),
    )


def test_strategy_time_series_valid_payload() -> None:
    dates = date_range_list("2024-01-01", "2024-01-03")
    data = pl.DataFrame(
        {
            "day_index": [0, 1, 2],
            "date": dates,
            "weight": [0.2, 0.3, 0.5],
            "price_usd": [42000.0, 43000.0, None],
            "locked": [True, True, False],
        }
    )
    series = StrategyTimeSeries(metadata=_metadata(), data=data)

    out = series.to_dataframe()
    assert list(out.columns) == ["day_index", "date", "weight", "price_usd", "locked"]
    assert np.isclose(float(out["weight"].sum()), 1.0)


def test_strategy_time_series_rejects_unknown_columns() -> None:
    dates = date_range_list("2024-01-01", "2024-01-02")
    data = pl.DataFrame(
        {
            "date": dates,
            "weight": [0.4, 0.6],
            "price_usd": [42000.0, 43000.0],
            "mystery": [1, 2],
        }
    )

    with pytest.raises(ValueError, match="Schema coverage missing"):
        StrategyTimeSeries(metadata=_metadata(), data=data)


def test_strategy_time_series_accepts_brk_passthrough_column() -> None:
    dates = date_range_list("2024-01-01", "2024-01-03")
    data = pl.DataFrame(
        {
            "date": dates,
            "weight": [0.2, 0.3, 0.5],
            "price_usd": [42000.0, 43000.0, None],
            "SplyCur": [100.0, 101.0, 102.0],
        }
    )
    series = StrategyTimeSeries(metadata=_metadata(), data=data)

    schema = series.schema()
    assert "SplyCur" in schema
    assert schema["SplyCur"].description == "BRK current circulating BTC supply."
    assert schema["SplyCur"].source == "brk"


def test_strategy_time_series_rejects_weight_sum_mismatch() -> None:
    dates = date_range_list("2024-01-01", "2024-01-02")
    data = pl.DataFrame(
        {
            "date": dates,
            "weight": [0.4, 0.4],
            "price_usd": [42000.0, 43000.0],
        }
    )

    with pytest.raises(ValueError, match="must sum to 1.0"):
        StrategyTimeSeries(
            metadata=StrategySeriesMetadata(
                strategy_id="test-strategy",
                strategy_version="1.2.3",
                run_id="run-1",
                config_hash="abc123",
                window_start=dt.datetime(2024, 1, 1),
                window_end=dt.datetime(2024, 1, 2),
            ),
            data=data,
        )


def test_strategy_time_series_batch_from_flat_dataframe() -> None:
    flat = pl.DataFrame(
        {
            "start_date": ["2024-01-01", "2024-01-01", "2024-02-01", "2024-02-01"],
            "end_date": ["2024-01-02", "2024-01-02", "2024-02-02", "2024-02-02"],
            "date": ["2024-01-01", "2024-01-02", "2024-02-01", "2024-02-02"],
            "weight": [0.45, 0.55, 0.4, 0.6],
            "price_usd": [40000.0, 41000.0, 50000.0, np.nan],
        }
    )
    batch = StrategyTimeSeriesBatch.from_flat_dataframe(
        flat,
        strategy_id="test-strategy",
        strategy_version="1.2.3",
        run_id="run-1",
        config_hash="abc123",
    )

    assert batch.window_count == 2
    assert batch.row_count == 4
    flattened = batch.to_dataframe()
    assert set(["start_date", "end_date", "date", "weight", "price_usd"]).issubset(
        flattened.columns
    )
    first = batch.for_window("2024-01-01", "2024-01-02")
    assert len(first.data) == 2


def test_strategy_time_series_batch_preserves_brk_passthrough_columns() -> None:
    flat = pl.DataFrame(
        {
            "start_date": ["2024-01-01", "2024-01-01"],
            "end_date": ["2024-01-02", "2024-01-02"],
            "date": ["2024-01-01", "2024-01-02"],
            "weight": [0.45, 0.55],
            "price_usd": [40000.0, 41000.0],
            "SplyCur": [100.0, 101.0],
        }
    )
    batch = StrategyTimeSeriesBatch.from_flat_dataframe(
        flat,
        strategy_id="test-strategy",
        strategy_version="1.2.3",
        run_id="run-1",
        config_hash="abc123",
    )

    flattened = batch.to_dataframe()
    assert "SplyCur" in flattened.columns


def test_strategy_time_series_batch_rejects_duplicate_windows() -> None:
    md = _metadata()
    dates = date_range_list("2024-01-01", "2024-01-03")
    data = pl.DataFrame(
        {
            "date": dates,
            "weight": [0.2, 0.3, 0.5],
            "price_usd": [1.0, 2.0, 3.0],
        }
    )
    window = StrategyTimeSeries(metadata=md, data=data)
    with pytest.raises(ValueError, match="Duplicate window key"):
        StrategyTimeSeriesBatch(
            strategy_id=md.strategy_id,
            strategy_version=md.strategy_version,
            run_id=md.run_id,
            config_hash=md.config_hash,
            windows=(window, window),
        )


def test_strategy_time_series_profile_returns_dataset_summary() -> None:
    dates = date_range_list("2024-01-01", "2024-01-03")
    data = pl.DataFrame(
        {
            "day_index": [0, 1, 2],
            "date": dates,
            "weight": [0.2, 0.3, 0.5],
            "price_usd": [42000.0, 43000.0, None],
            "SplyCur": [100.0, None, 102.0],
        }
    )
    series = StrategyTimeSeries(metadata=_metadata(), data=data)

    profile = series.profile()

    assert profile["row_count"] == 3
    assert profile["column_count"] == 5
    assert profile["date_start"] == "2024-01-01T00:00:00"
    assert profile["date_end"] == "2024-01-03T00:00:00"
    assert "weight" in profile["numeric_columns"]
    assert profile["columns_profile"]["price_usd"]["null_count"] == 1
    assert profile["columns_profile"]["SplyCur"]["numeric_summary"]["count"] == 2


def test_strategy_time_series_weight_diagnostics_returns_expected_metrics() -> None:
    dates = date_range_list("2024-01-01", "2024-01-03")
    data = pl.DataFrame(
        {
            "date": dates,
            "weight": [0.2, 0.3, 0.5],
            "price_usd": [100.0, 110.0, 120.0],
        }
    )
    series = StrategyTimeSeries(metadata=_metadata(), data=data)

    diagnostics = series.weight_diagnostics(top_k=2)

    assert diagnostics["sample_size"] == 3
    assert np.isclose(diagnostics["sum"], 1.0)
    assert np.isclose(diagnostics["hhi"], 0.38)
    assert np.isclose(diagnostics["effective_n"], 1.0 / 0.38)
    assert len(diagnostics["top_weights"]) == 2
    assert diagnostics["top_weights"][0]["date"] == "2024-01-03T00:00:00"
    assert np.isclose(diagnostics["top_weights"][0]["weight"], 0.5)


def test_strategy_time_series_returns_diagnostics_returns_expected_metrics() -> None:
    dates = date_range_list("2024-01-01", "2024-01-03")
    data = pl.DataFrame(
        {
            "date": dates,
            "weight": [0.2, 0.3, 0.5],
            "price_usd": [100.0, 110.0, 121.0],
        }
    )
    series = StrategyTimeSeries(metadata=_metadata(), data=data)

    diagnostics = series.returns_diagnostics()

    assert diagnostics["price_observations"] == 3
    assert diagnostics["return_observations"] == 2
    assert np.isclose(diagnostics["cumulative_return"], 0.21)
    assert np.isclose(diagnostics["mean_simple_return"], 0.1)
    assert diagnostics["best_day_date"] == "2024-01-02T00:00:00"
    assert diagnostics["worst_day_date"] == "2024-01-02T00:00:00"
    assert np.isclose(diagnostics["max_drawdown"], 0.0)


def test_strategy_time_series_outlier_report_detects_mad_outliers() -> None:
    dates = date_range_list("2024-01-01", "2024-01-05")
    data = pl.DataFrame(
        {
            "date": dates,
            "weight": [0.05, 0.1, 0.1, 0.15, 0.6],
            "price_usd": [100.0, 101.0, 102.0, 103.0, 1000.0],
            "SplyCur": [10.0, 10.0, 10.0, 10.0, 10.0],
        }
    )
    series = StrategyTimeSeries(
        metadata=StrategySeriesMetadata(
            strategy_id="test-strategy",
            strategy_version="1.2.3",
            run_id="run-1",
            config_hash="abc123",
            window_start=dt.datetime(2024, 1, 1),
            window_end=dt.datetime(2024, 1, 5),
        ),
        data=data,
    )

    report = series.outlier_report(columns=["price_usd"], method="mad", threshold=3.5)

    assert list(report.columns) == ["date", "column", "value", "score", "method", "threshold"]
    assert len(report) == 1
    row = report.row(0)
    col_idx = report.columns.index("column")
    date_idx = report.columns.index("date")
    value_idx = report.columns.index("value")
    method_idx = report.columns.index("method")
    assert row[col_idx] == "price_usd"
    assert str(row[date_idx])[:10] == "2024-01-05"
    assert np.isclose(float(row[value_idx]), 1000.0)
    assert row[method_idx] == "mad"


def test_strategy_time_series_rolling_statistics_returns_expected_columns_and_values() -> None:
    dates = date_range_list("2024-01-01", "2024-01-04")
    data = pl.DataFrame(
        {
            "date": dates,
            "weight": [0.1, 0.2, 0.3, 0.4],
            "price_usd": [100.0, 110.0, 120.0, 130.0],
        }
    )
    series = StrategyTimeSeries(
        metadata=StrategySeriesMetadata(
            strategy_id="test-strategy",
            strategy_version="1.2.3",
            run_id="run-1",
            config_hash="abc123",
            window_start=dt.datetime(2024, 1, 1),
            window_end=dt.datetime(2024, 1, 4),
        ),
        data=data,
    )

    stats = series.rolling_statistics(windows=(2,))

    assert "price_usd_mean_2" in stats.columns
    assert "return_std_2" in stats.columns
    assert np.isclose(float(stats["price_usd_mean_2"][1]), 105.0)
    assert np.isclose(float(stats["price_usd_mean_2"][3]), 125.0)


def test_strategy_time_series_autocorrelation_returns_expected_shape() -> None:
    dates = date_range_list("2024-01-01", "2024-01-05")
    data = pl.DataFrame(
        {
            "date": dates,
            "weight": [0.1, 0.2, 0.2, 0.2, 0.3],
            "price_usd": [100.0, 110.0, 100.0, 110.0, 100.0],
        }
    )
    series = StrategyTimeSeries(
        metadata=StrategySeriesMetadata(
            strategy_id="test-strategy",
            strategy_version="1.2.3",
            run_id="run-1",
            config_hash="abc123",
            window_start=dt.datetime(2024, 1, 1),
            window_end=dt.datetime(2024, 1, 5),
        ),
        data=data,
    )

    acf = series.autocorrelation(lags=(1, 2), series="returns")

    assert acf["series"] == "returns"
    assert acf["lags"] == [1, 2]
    assert acf["observations"] == 4
    assert set(acf["autocorrelation"].keys()) == {"1", "2"}
    assert acf["autocorrelation"]["1"] is not None


def test_strategy_time_series_drawdown_table_returns_recovered_episode() -> None:
    dates = date_range_list("2024-01-01", "2024-01-04")
    data = pl.DataFrame(
        {
            "date": dates,
            "weight": [0.1, 0.2, 0.3, 0.4],
            "price_usd": [100.0, 90.0, 95.0, 100.0],
        }
    )
    series = StrategyTimeSeries(
        metadata=StrategySeriesMetadata(
            strategy_id="test-strategy",
            strategy_version="1.2.3",
            run_id="run-1",
            config_hash="abc123",
            window_start=dt.datetime(2024, 1, 1),
            window_end=dt.datetime(2024, 1, 4),
        ),
        data=data,
    )

    drawdowns = series.drawdown_table(top_n=3)

    assert len(drawdowns) == 1
    row0 = drawdowns.row(0, named=True)
    assert np.isclose(float(row0["max_drawdown"]), -0.1)
    assert bool(row0["recovered"]) is True
    assert str(row0["peak_date"])[:10] == "2024-01-01"
    assert str(row0["recovery_date"])[:10] == "2024-01-04"


def test_strategy_time_series_seasonality_profile_weekday_returns_expected_counts() -> None:
    dates = date_range_list("2024-01-01", "2024-01-07")
    data = pl.DataFrame(
        {
            "date": dates,
            "weight": [1 / 7] * 7,
            "price_usd": [100.0, 101.0, 99.0, 102.0, 100.0, 103.0, 104.0],
        }
    )
    series = StrategyTimeSeries(
        metadata=StrategySeriesMetadata(
            strategy_id="test-strategy",
            strategy_version="1.2.3",
            run_id="run-1",
            config_hash="abc123",
            window_start=dt.datetime(2024, 1, 1),
            window_end=dt.datetime(2024, 1, 7),
        ),
        data=data,
    )

    profile = series.seasonality_profile(freq="weekday", series="returns")

    assert len(profile) == 7
    monday = profile.filter(pl.col("period_label") == "Mon").row(0, named=True)
    assert int(monday["count"]) == 0
    tuesday = profile.filter(pl.col("period_label") == "Tue").row(0, named=True)
    assert int(tuesday["count"]) == 1


def test_strategy_time_series_resample_returns_expected_frequency_shape() -> None:
    dates = date_range_list("2024-01-01", "2024-01-10")
    data = pl.DataFrame(
        {
            "date": dates,
            "weight": [0.1] * 10,
            "price_usd": np.arange(100.0, 110.0),
        }
    )
    series = StrategyTimeSeries(
        metadata=StrategySeriesMetadata(
            strategy_id="test-strategy",
            strategy_version="1.2.3",
            run_id="run-1",
            config_hash="abc123",
            window_start=dt.datetime(2024, 1, 1),
            window_end=dt.datetime(2024, 1, 10),
        ),
        data=data,
    )

    weekly = series.resample("W", agg="mean")

    assert "date" in weekly.columns
    assert "price_usd" in weekly.columns
    assert len(weekly) >= 2


def test_strategy_time_series_decompose_additive_returns_expected_columns() -> None:
    dates = date_range_list("2024-01-01", "2024-01-12")
    data = pl.DataFrame(
        {
            "date": dates,
            "weight": [1 / 12] * 12,
            "price_usd": [100.0, 102.0, 104.0, 103.0, 105.0, 107.0, 106.0, 108.0, 110.0, 109.0, 111.0, 113.0],
        }
    )
    series = StrategyTimeSeries(
        metadata=StrategySeriesMetadata(
            strategy_id="test-strategy",
            strategy_version="1.2.3",
            run_id="run-1",
            config_hash="abc123",
            window_start=dt.datetime(2024, 1, 1),
            window_end=dt.datetime(2024, 1, 12),
        ),
        data=data,
    )

    decomposition = series.decompose(period=3, model="additive", series="price")

    assert set(["observed", "trend", "seasonal", "residual", "model", "period", "series"]).issubset(
        decomposition.columns
    )
    assert decomposition["model"][0] == "additive"


def test_strategy_time_series_detrend_linear_returns_detrended_columns() -> None:
    dates = date_range_list("2024-01-01", "2024-01-05")
    data = pl.DataFrame(
        {
            "date": dates,
            "weight": [0.1, 0.15, 0.2, 0.25, 0.3],
            "price_usd": [100.0, 101.0, 102.0, 103.0, 104.0],
        }
    )
    series = StrategyTimeSeries(
        metadata=StrategySeriesMetadata(
            strategy_id="test-strategy",
            strategy_version="1.2.3",
            run_id="run-1",
            config_hash="abc123",
            window_start=dt.datetime(2024, 1, 1),
            window_end=dt.datetime(2024, 1, 5),
        ),
        data=data,
    )

    detrended = series.detrend(method="linear")

    assert "price_usd_detrended" in detrended.columns
    assert "weight_detrended" in detrended.columns
    assert (detrended["price_usd_detrended"].is_not_null() & ~detrended["price_usd_detrended"].is_nan()).any()


def test_strategy_time_series_difference_returns_expected_shape() -> None:
    dates = date_range_list("2024-01-01", "2024-01-06")
    data = pl.DataFrame(
        {
            "date": dates,
            "weight": [0.1, 0.15, 0.2, 0.2, 0.15, 0.2],
            "price_usd": [100.0, 103.0, 106.0, 109.0, 112.0, 115.0],
        }
    )
    series = StrategyTimeSeries(
        metadata=StrategySeriesMetadata(
            strategy_id="test-strategy",
            strategy_version="1.2.3",
            run_id="run-1",
            config_hash="abc123",
            window_start=dt.datetime(2024, 1, 1),
            window_end=dt.datetime(2024, 1, 6),
        ),
        data=data,
    )

    diffed = series.difference(order=1)

    assert "price_usd_diff" in diffed.columns
    first_val = diffed["price_usd_diff"][0]
    assert first_val is None or (isinstance(first_val, float) and np.isnan(first_val))
    assert np.isclose(float(diffed["price_usd_diff"][2]), 3.0)


def test_strategy_time_series_acf_pacf_returns_expected_columns() -> None:
    dates = date_range_list("2024-01-01", "2024-01-08")
    data = pl.DataFrame(
        {
            "date": dates,
            "weight": [0.05, 0.1, 0.15, 0.2, 0.15, 0.1, 0.1, 0.15],
            "price_usd": [100.0, 102.0, 101.0, 103.0, 102.0, 104.0, 103.0, 105.0],
        }
    )
    series = StrategyTimeSeries(
        metadata=StrategySeriesMetadata(
            strategy_id="test-strategy",
            strategy_version="1.2.3",
            run_id="run-1",
            config_hash="abc123",
            window_start=dt.datetime(2024, 1, 1),
            window_end=dt.datetime(2024, 1, 8),
        ),
        data=data,
    )

    diagnostics = series.acf_pacf(lags=3, series="returns")

    assert list(diagnostics["lag"]) == [1, 2, 3]
    assert set(["acf", "pacf", "series", "observations"]).issubset(diagnostics.columns)


def test_strategy_time_series_cross_correlation_returns_lag_window() -> None:
    dates = date_range_list("2024-01-01", "2024-01-08")
    data = pl.DataFrame(
        {
            "date": dates,
            "weight": [0.05, 0.1, 0.15, 0.2, 0.15, 0.1, 0.1, 0.15],
            "price_usd": [100.0, 101.0, 102.0, 103.0, 102.0, 101.0, 102.0, 103.0],
        }
    )
    series = StrategyTimeSeries(
        metadata=StrategySeriesMetadata(
            strategy_id="test-strategy",
            strategy_version="1.2.3",
            run_id="run-1",
            config_hash="abc123",
            window_start=dt.datetime(2024, 1, 1),
            window_end=dt.datetime(2024, 1, 8),
        ),
        data=data,
    )

    ccf = series.cross_correlation("price", max_lag=2, series="returns")

    assert list(ccf["lag"]) == [-2, -1, 0, 1, 2]
    assert "correlation" in ccf.columns
    assert "observations" in ccf.columns


def test_strategy_time_series_spectral_density_periodogram_returns_expected_columns() -> None:
    dates = date_range_list("2024-01-01", "2024-01-16")
    data = pl.DataFrame(
        {
            "date": dates,
            "weight": [1 / 16] * 16,
            "price_usd": [100.0, 101.0, 100.0, 99.0, 100.0, 101.0, 100.0, 99.0, 100.0, 101.0, 100.0, 99.0, 100.0, 101.0, 100.0, 99.0],
        }
    )
    series = StrategyTimeSeries(
        metadata=StrategySeriesMetadata(
            strategy_id="test-strategy",
            strategy_version="1.2.3",
            run_id="run-1",
            config_hash="abc123",
            window_start=dt.datetime(2024, 1, 1),
            window_end=dt.datetime(2024, 1, 16),
        ),
        data=data,
    )

    spectrum = series.spectral_density(series="price")

    assert set(["frequency", "power", "series", "method", "observations"]).issubset(spectrum.columns)
    assert spectrum["method"][0] == "periodogram"


def test_strategy_time_series_integration_order_returns_per_column_output() -> None:
    weights = np.linspace(1.0, 2.0, 10)
    weights = weights / weights.sum()
    dates = date_range_list("2024-01-01", "2024-01-10")
    data = pl.DataFrame(
        {
            "date": dates,
            "weight": weights,
            "price_usd": np.linspace(100.0, 110.0, 10),
            "SplyCur": np.linspace(10.0, 11.0, 10),
        }
    )
    series = StrategyTimeSeries(
        metadata=StrategySeriesMetadata(
            strategy_id="test-strategy",
            strategy_version="1.2.3",
            run_id="run-1",
            config_hash="abc123",
            window_start=dt.datetime(2024, 1, 1),
            window_end=dt.datetime(2024, 1, 10),
        ),
        data=data,
    )

    integration = series.integration_order(max_order=2)

    assert set(["column", "integration_order", "detected", "method"]).issubset(integration.columns)
    assert "price_usd" in set(integration["column"])
    assert "SplyCur" in set(integration["column"])
