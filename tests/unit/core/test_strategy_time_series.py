from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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
        window_start=pd.Timestamp("2024-01-01"),
        window_end=pd.Timestamp("2024-01-03"),
    )


def test_strategy_time_series_valid_payload() -> None:
    data = pd.DataFrame(
        {
            "day_index": [0, 1, 2],
            "date": pd.date_range("2024-01-01", periods=3, freq="D"),
            "weight": [0.2, 0.3, 0.5],
            "price_usd": [42000.0, 43000.0, np.nan],
            "locked": [True, True, False],
        }
    )
    series = StrategyTimeSeries(metadata=_metadata(), data=data)

    out = series.to_dataframe()
    assert list(out.columns) == ["day_index", "date", "weight", "price_usd", "locked"]
    assert np.isclose(float(out["weight"].sum()), 1.0)


def test_strategy_time_series_rejects_unknown_columns() -> None:
    data = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=2, freq="D"),
            "weight": [0.4, 0.6],
            "price_usd": [42000.0, 43000.0],
            "mystery": [1, 2],
        }
    )

    with pytest.raises(ValueError, match="Schema coverage missing"):
        StrategyTimeSeries(metadata=_metadata(), data=data)


def test_strategy_time_series_accepts_coinmetrics_passthrough_column() -> None:
    data = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3, freq="D"),
            "weight": [0.2, 0.3, 0.5],
            "price_usd": [42000.0, 43000.0, np.nan],
            "SplyCur": [100.0, 101.0, 102.0],
        }
    )
    series = StrategyTimeSeries(metadata=_metadata(), data=data)

    schema = series.schema()
    assert "SplyCur" in schema
    assert schema["SplyCur"].description == "CoinMetrics current circulating BTC supply."
    assert schema["SplyCur"].source == "coinmetrics"


def test_strategy_time_series_rejects_weight_sum_mismatch() -> None:
    data = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=2, freq="D"),
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
                window_start=pd.Timestamp("2024-01-01"),
                window_end=pd.Timestamp("2024-01-02"),
            ),
            data=data,
        )


def test_strategy_time_series_batch_from_flat_dataframe() -> None:
    flat = pd.DataFrame(
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


def test_strategy_time_series_batch_preserves_coinmetrics_passthrough_columns() -> None:
    flat = pd.DataFrame(
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
    data = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3, freq="D"),
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
    data = pd.DataFrame(
        {
            "day_index": [0, 1, 2],
            "date": pd.date_range("2024-01-01", periods=3, freq="D"),
            "weight": [0.2, 0.3, 0.5],
            "price_usd": [42000.0, 43000.0, np.nan],
            "SplyCur": [100.0, np.nan, 102.0],
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
    data = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3, freq="D"),
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
    data = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3, freq="D"),
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
    data = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5, freq="D"),
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
            window_start=pd.Timestamp("2024-01-01"),
            window_end=pd.Timestamp("2024-01-05"),
        ),
        data=data,
    )

    report = series.outlier_report(columns=["price_usd"], method="mad", threshold=3.5)

    assert list(report.columns) == ["date", "column", "value", "score", "method", "threshold"]
    assert len(report) == 1
    assert report.iloc[0]["column"] == "price_usd"
    assert report.iloc[0]["date"] == pd.Timestamp("2024-01-05")
    assert np.isclose(report.iloc[0]["value"], 1000.0)
    assert report.iloc[0]["method"] == "mad"


def test_strategy_time_series_rolling_statistics_returns_expected_columns_and_values() -> None:
    data = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4, freq="D"),
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
            window_start=pd.Timestamp("2024-01-01"),
            window_end=pd.Timestamp("2024-01-04"),
        ),
        data=data,
    )

    stats = series.rolling_statistics(windows=(2,))

    assert "price_usd_mean_2" in stats.columns
    assert "return_std_2" in stats.columns
    assert np.isclose(stats.loc[1, "price_usd_mean_2"], 105.0)
    assert np.isclose(stats.loc[3, "price_usd_mean_2"], 125.0)


def test_strategy_time_series_autocorrelation_returns_expected_shape() -> None:
    data = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5, freq="D"),
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
            window_start=pd.Timestamp("2024-01-01"),
            window_end=pd.Timestamp("2024-01-05"),
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
    data = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4, freq="D"),
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
            window_start=pd.Timestamp("2024-01-01"),
            window_end=pd.Timestamp("2024-01-04"),
        ),
        data=data,
    )

    drawdowns = series.drawdown_table(top_n=3)

    assert len(drawdowns) == 1
    assert np.isclose(float(drawdowns.loc[0, "max_drawdown"]), -0.1)
    assert bool(drawdowns.loc[0, "recovered"]) is True
    assert drawdowns.loc[0, "peak_date"] == pd.Timestamp("2024-01-01")
    assert drawdowns.loc[0, "recovery_date"] == pd.Timestamp("2024-01-04")


def test_strategy_time_series_seasonality_profile_weekday_returns_expected_counts() -> None:
    data = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=7, freq="D"),
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
            window_start=pd.Timestamp("2024-01-01"),
            window_end=pd.Timestamp("2024-01-07"),
        ),
        data=data,
    )

    profile = series.seasonality_profile(freq="weekday", series="returns")

    assert len(profile) == 7
    monday = profile.loc[profile["period_label"] == "Mon"].iloc[0]
    assert int(monday["count"]) == 0
    tuesday = profile.loc[profile["period_label"] == "Tue"].iloc[0]
    assert int(tuesday["count"]) == 1
