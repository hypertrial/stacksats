from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from stacksats.strategy_time_series import StrategySeriesMetadata, StrategyTimeSeries, StrategyTimeSeriesBatch


def _metadata(
    *,
    strategy_id: str = "s1",
    strategy_version: str = "1.0.0",
    run_id: str = "run-1",
    config_hash: str = "cfg-1",
    schema_version: str = "1.0.0",
    window_start: dt.datetime | None = dt.datetime(2024, 1, 1),
    window_end: dt.datetime | None = dt.datetime(2024, 1, 2),
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


def _valid_data() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2)],
            "weight": [0.4, 0.6],
            "price_usd": [40000.0, 41000.0],
        }
    )


def _window(md: StrategySeriesMetadata | None = None) -> StrategyTimeSeries:
    return StrategyTimeSeries(metadata=md or _metadata(), data=_valid_data())


def test_strategy_time_series_requires_polars_dataframe_data() -> None:
    with pytest.raises(TypeError, match="Polars DataFrame"):
        StrategyTimeSeries(metadata=_metadata(), data=[1, 2])  # type: ignore[arg-type]


def test_strategy_time_series_rejects_missing_required_columns() -> None:
    missing_price = pl.DataFrame(
        {
            "date": [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2)],
            "weight": [0.4, 0.6],
        }
    )
    with pytest.raises(ValueError, match="missing required columns"):
        StrategyTimeSeries(metadata=_metadata(), data=missing_price)


def test_strategy_time_series_rejects_invalid_dates() -> None:
    bad_date = pl.DataFrame(
        {
            "date": ["2024-01-01", "not-a-date"],
            "weight": [0.4, 0.6],
            "price_usd": [40000.0, 41000.0],
        }
    )
    with pytest.raises(Exception, match="valid datetimes|conversion|cast|datetime"):
        StrategyTimeSeries(metadata=_metadata(), data=bad_date)


def test_strategy_time_series_rejects_duplicate_dates() -> None:
    duplicated = pl.DataFrame(
        {
            "date": [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 1)],
            "weight": [0.4, 0.6],
            "price_usd": [40000.0, 41000.0],
        }
    )
    with pytest.raises(ValueError, match="must not contain duplicates"):
        StrategyTimeSeries(metadata=_metadata(), data=duplicated)


def test_strategy_time_series_data_returns_copy() -> None:
    series = _window()
    copied = series.data.with_columns(pl.lit(dt.datetime(2024, 1, 3)).alias("date"))
    assert copied["date"][0] == dt.datetime(2024, 1, 3)

    orig_dates = series.to_dataframe()["date"].to_list()
    assert orig_dates == [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2)]


def test_strategy_time_series_rejects_missing_interior_dates_with_bounds() -> None:
    missing_gap = pl.DataFrame(
        {
            "date": [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 3)],
            "weight": [0.4, 0.6],
            "price_usd": [40000.0, 41000.0],
        }
    )
    with pytest.raises(ValueError, match="must exactly match the daily range"):
        StrategyTimeSeries(
            metadata=_metadata(
                window_start=dt.datetime(2024, 1, 1),
                window_end=dt.datetime(2024, 1, 3),
            ),
            data=missing_gap,
        )


def test_strategy_time_series_allows_non_daily_dates_without_bounds() -> None:
    sparse = pl.DataFrame(
        {
            "date": [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 3)],
            "weight": [0.4, 0.6],
            "price_usd": [40000.0, 41000.0],
        }
    )
    series = StrategyTimeSeries(
        metadata=_metadata(window_start=None, window_end=None),
        data=sparse,
    )
    assert series.row_count == 2


def test_strategy_time_series_rejects_window_start_mismatch() -> None:
    with pytest.raises(ValueError, match="start date does not match metadata.window_start"):
        StrategyTimeSeries(
            metadata=_metadata(window_start=dt.datetime(2024, 1, 2), window_end=dt.datetime(2024, 1, 2)),
            data=_valid_data(),
        )


def test_strategy_time_series_rejects_window_end_mismatch() -> None:
    with pytest.raises(ValueError, match="end date does not match metadata.window_end"):
        StrategyTimeSeries(
            metadata=_metadata(window_start=dt.datetime(2024, 1, 1), window_end=dt.datetime(2024, 1, 3)),
            data=_valid_data(),
        )


def test_strategy_time_series_rejects_nonfinite_weights() -> None:
    nonfinite_weights = _valid_data().with_columns(pl.Series("weight", [0.4, float("inf")]))
    with pytest.raises(ValueError, match="must contain finite numeric values"):
        StrategyTimeSeries(metadata=_metadata(), data=nonfinite_weights)


def test_strategy_time_series_rejects_negative_weights() -> None:
    negative_weights = _valid_data().with_columns(pl.Series("weight", [-0.1, 1.1]))
    with pytest.raises(ValueError, match="must not contain negative values"):
        StrategyTimeSeries(metadata=_metadata(), data=negative_weights)


def test_strategy_time_series_rejects_nonnumeric_price_usd() -> None:
    bad_price = pl.DataFrame(
        {
            "date": [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2)],
            "weight": [0.4, 0.6],
            "price_usd": ["40000.0", "bad"],
        }
    )
    with pytest.raises(ValueError, match="numeric when present"):
        StrategyTimeSeries(metadata=_metadata(), data=bad_price)


def test_strategy_time_series_rejects_nonfinite_price_usd() -> None:
    nonfinite_price = _valid_data().with_columns(pl.Series("price_usd", [40000.0, float("inf")]))
    with pytest.raises(ValueError, match="price_usd' must be finite"):
        StrategyTimeSeries(metadata=_metadata(), data=nonfinite_price)


def test_strategy_time_series_rejects_nonboolean_locked_column() -> None:
    bad_locked = _valid_data().with_columns(pl.Series("locked", [True, "bad"], strict=False))
    with pytest.raises(ValueError, match="boolean values"):
        StrategyTimeSeries(metadata=_metadata(), data=bad_locked)


def test_strategy_time_series_rejects_nonnumeric_day_index() -> None:
    bad_day_index = _valid_data().with_columns(pl.Series("day_index", [0, "bad"], strict=False))
    with pytest.raises(ValueError, match="integer values"):
        StrategyTimeSeries(metadata=_metadata(), data=bad_day_index)


def test_strategy_time_series_rejects_negative_day_index() -> None:
    negative_day_index = _valid_data().with_columns(pl.Series("day_index", [-1, 0]))
    with pytest.raises(ValueError, match="day_index' must be >= 0"):
        StrategyTimeSeries(metadata=_metadata(), data=negative_day_index)


def test_strategy_time_series_rejects_noncontiguous_day_index() -> None:
    noncontiguous_day_index = _valid_data().with_columns(pl.Series("day_index", [0, 2]))
    with pytest.raises(ValueError, match="must be contiguous starting at 0"):
        StrategyTimeSeries(metadata=_metadata(), data=noncontiguous_day_index)


def test_batch_requires_polars_dataframe_and_columns() -> None:
    with pytest.raises(TypeError, match="Polars DataFrame"):
        StrategyTimeSeriesBatch.from_flat_dataframe(  # type: ignore[arg-type]
            [{"a": 1}],
            strategy_id="s1",
            strategy_version="1.0.0",
            run_id="run-1",
            config_hash="cfg-1",
        )

    with pytest.raises(ValueError, match="Flat dataframe missing required columns"):
        StrategyTimeSeriesBatch.from_flat_dataframe(
            pl.DataFrame(
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


def test_batch_validate_mismatch_and_window_lookup_paths() -> None:
    with pytest.raises(ValueError, match="strategy_id does not match"):
        StrategyTimeSeriesBatch(
            strategy_id="other",
            strategy_version="1.0.0",
            run_id="run-1",
            config_hash="cfg-1",
            windows=(_window(_metadata(strategy_id="s1")),),
        )

    with pytest.raises(ValueError, match="must define metadata.window_start and metadata.window_end"):
        StrategyTimeSeriesBatch(
            strategy_id="s1",
            strategy_version="1.0.0",
            run_id="run-1",
            config_hash="cfg-1",
            windows=(_window(_metadata(window_start=None, window_end=None)),),
        )

    batch = StrategyTimeSeriesBatch(
        strategy_id="s1",
        strategy_version="1.0.0",
        run_id="run-1",
        config_hash="cfg-1",
        windows=(_window(),),
    )
    assert len(list(batch.iter_windows())) == 1
    with pytest.raises(KeyError, match="Window not found"):
        batch.for_window("2030-01-01", "2030-01-02")


def test_native_helpers_and_diagnostic_edge_paths() -> None:
    assert StrategyTimeSeries._native_float(None) is None
    assert StrategyTimeSeries._native_float(float("inf")) is None
    assert StrategyTimeSeries._native_timestamp(None) is None

    summary = StrategyTimeSeries._series_numeric_summary(
        pl.Series("s", [None, None], dtype=pl.Float64)
    )
    assert summary["count"] == 0
    assert summary["mean"] is None

    series = StrategyTimeSeries(
        metadata=StrategySeriesMetadata(
            strategy_id="test-strategy",
            strategy_version="1.2.3",
            run_id="run-1",
            config_hash="abc123",
            window_start=dt.datetime(2024, 1, 1),
            window_end=dt.datetime(2024, 1, 3),
        ),
        data=pl.DataFrame(
            {
                "date": [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2), dt.datetime(2024, 1, 3)],
                "weight": [0.2, 0.3, 0.5],
                "price_usd": [100.0, 110.0, 120.0],
                "SplyCur": [10.0, 10.0, 10.0],
            }
        ),
    )

    weightless = series.to_dataframe().with_columns(pl.lit(float("nan")).alias("weight"))
    object.__setattr__(series, "_data", weightless)
    weight_diag = series.weight_diagnostics()
    assert weight_diag["sample_size"] == 0

    no_price = series.to_dataframe().with_columns(pl.lit(float("nan")).alias("price_usd"))
    object.__setattr__(series, "_data", no_price)
    ret_diag = series.returns_diagnostics()
    assert ret_diag["price_observations"] == 0

    short_metric = series.to_dataframe().with_columns(
        pl.Series("SplyCur", [None, None, 10.0], dtype=pl.Float64)
    )
    object.__setattr__(series, "_data", short_metric)
    assert series.outlier_report(columns=["SplyCur"], method="mad").is_empty()

    flat_metric = series.to_dataframe().with_columns(pl.Series("SplyCur", [10.0, 10.0, 10.0]))
    object.__setattr__(series, "_data", flat_metric)
    assert series.outlier_report(columns=["SplyCur"], method="mad").is_empty()

    outlier_price = series.to_dataframe().with_columns(
        pl.Series("price_usd", [100.0, 101.0, 1000.0])
    )
    object.__setattr__(series, "_data", outlier_price)
    out_z = series.outlier_report(columns=["price_usd"], method="zscore", threshold=0.5)
    assert not out_z.is_empty()
    assert set(out_z["method"].to_list()) == {"zscore"}


def test_outlier_and_returns_diagnostics_validation_paths() -> None:
    series = _window()
    with pytest.raises(ValueError, match="method must be one of"):
        series.outlier_report(method="bogus")
    with pytest.raises(ValueError, match="Unknown columns"):
        series.outlier_report(columns=["unknown_col"])
    with pytest.raises(ValueError, match="threshold must be > 0"):
        series.outlier_report(threshold=0.0)

    constant = StrategyTimeSeries(
        metadata=_metadata(window_end=dt.datetime(2024, 1, 3)),
        data=pl.DataFrame(
            {
                "date": [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2), dt.datetime(2024, 1, 3)],
                "weight": [1 / 3, 1 / 3, 1 / 3],
                "price_usd": [100.0, 100.0, 100.0],
                "SplyCur": [10.0, 10.0, 10.0],
            }
        ),
    )
    report = constant.outlier_report(method="zscore")
    assert report.is_empty()
    assert list(report.columns) == ["date", "column", "value", "score", "method", "threshold"]

    iqr = StrategyTimeSeries(
        metadata=_metadata(window_end=dt.datetime(2024, 1, 5)),
        data=pl.DataFrame(
            {
                "date": [
                    dt.datetime(2024, 1, 1),
                    dt.datetime(2024, 1, 2),
                    dt.datetime(2024, 1, 3),
                    dt.datetime(2024, 1, 4),
                    dt.datetime(2024, 1, 5),
                ],
                "weight": [0.2] * 5,
                "price_usd": [100.0, 101.0, 102.0, 103.0, 1000.0],
            }
        ),
    )
    iqr_report = iqr.outlier_report(method="IQR")
    assert not iqr_report.is_empty()

    short_prices = StrategyTimeSeries(
        metadata=_metadata(),
        data=pl.DataFrame(
            {
                "date": [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2)],
                "weight": [0.4, 0.6],
                "price_usd": [40000.0, None],
            }
        ),
    )
    diagnostics = short_prices.returns_diagnostics()
    assert diagnostics["price_observations"] == 1
    assert diagnostics["return_observations"] == 0
    assert diagnostics["cumulative_return"] is None
    assert diagnostics["std_simple_return"] is None
    assert diagnostics["annualized_volatility"] is None
