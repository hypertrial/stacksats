from __future__ import annotations

import datetime as dt
import warnings

import numpy as np
import polars as pl
import pytest

from stacksats.strategy_time_series import StrategySeriesMetadata, WeightTimeSeries
from stacksats.strategy_time_series_analysis import _autocorr_pl


def _series(prices: list[float] | None = None) -> WeightTimeSeries:
    price_vals = prices or [100.0, 95.0, 90.0, 85.0, 80.0, 82.0]
    dates = [
        dt.datetime(2024, 1, 1) + dt.timedelta(days=offset)
        for offset in range(len(price_vals))
    ]
    data = pl.DataFrame(
        {
            "date": dates,
            "weight": [1.0 / len(price_vals)] * len(price_vals),
            "price_usd": price_vals,
            "mvrv": [float(offset) for offset in range(len(price_vals))],
        }
    )
    meta = StrategySeriesMetadata(
        strategy_id="s1",
        strategy_version="1.0.0",
        run_id="run-1",
        config_hash="cfg",
        schema_version="1.0.0",
        window_start=dates[0],
        window_end=dates[-1],
    )
    return WeightTimeSeries(metadata=meta, data=data)


def test_eda_series_variants_and_errors() -> None:
    ts = _series([100.0, 0.0, -1.0, 3.0, 4.0, 5.0])

    log_returns = ts._eda_value_series("log_returns")
    assert log_returns.len() == ts.data.height

    weight = ts._eda_value_series("weight")
    assert np.isfinite(weight.to_numpy().astype(float)).all()

    with pytest.raises(ValueError, match="series must be one of"):
        ts._eda_value_series("bad-series")


def test_analysis_helper_validation_edges() -> None:
    ts = _series([100.0, 101.0, 102.0])
    assert _autocorr_pl(pl.Series("x", [1.0]), 1) is None

    with pytest.raises(ValueError, match="Unknown price column"):
        ts._eda_price_series("missing")
    with pytest.raises(ValueError, match="positive integers"):
        ts._normalize_positive_ints([1, 0], "lags")
    with pytest.raises(ValueError, match="lags must be > 0"):
        ts._resolve_lags(0)

    acf = ts.autocorrelation(lags=(10,), series="returns")
    assert acf["autocorrelation"]["10"] is None


def test_drawdown_branches_no_valid_and_unrecovered() -> None:
    ts = _series()
    empty_df = ts.to_dataframe().with_columns(pl.lit(float("nan")).alias("price_usd"))
    empty = WeightTimeSeries(metadata=ts.metadata, data=empty_df).drawdown_table()
    assert empty.is_empty()

    unrecovered = _series([100.0, 90.0, 80.0, 70.0, 60.0, 50.0]).drawdown_table(top_n=3)
    assert not unrecovered.is_empty()
    assert bool((unrecovered["recovered"] == False).any())  # noqa: E712

    flat = _series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0]).drawdown_table(top_n=3)
    assert flat.is_empty()

    with pytest.raises(ValueError, match="top_n must be > 0"):
        _series().drawdown_table(top_n=0)


def test_resolve_series_like_and_cross_correlation_paths() -> None:
    ts = _series()
    short_other = pl.Series("other", [1.0, 2.0])

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        corr = ts.cross_correlation(short_other, max_lag=2, series="returns")
        assert not corr.is_empty()

        corr_col = ts.cross_correlation("mvrv", max_lag=1, series="returns")
        assert not corr_col.is_empty()


def test_pacf_edge_paths() -> None:
    values = pl.Series("constant", [1.0, 1.0, 1.0, 1.0])
    assert WeightTimeSeries._pacf_at_lag(values, 0) is None
    assert WeightTimeSeries._pacf_at_lag(values.head(2), 2) is None
    assert WeightTimeSeries._pacf_at_lag(values, 2) is None


def test_resample_and_decompose_error_paths() -> None:
    ts = _series()
    non_numeric = ts.to_dataframe().with_columns(
        pl.col("weight").cast(pl.Utf8).alias("weight"),
        pl.col("price_usd").cast(pl.Utf8).alias("price_usd"),
        pl.col("mvrv").cast(pl.Utf8).alias("mvrv"),
    )
    object.__setattr__(ts, "_data", non_numeric)
    out = ts.resample("D")
    assert list(out.columns) == ["date"]
    assert list(_series().resample("D").columns) == ["date", "weight", "price_usd", "mvrv"]
    assert list(_series().resample("W").columns) == ["date", "weight", "price_usd", "mvrv"]
    assert list(_series().resample("M", agg="sum").columns) == ["date", "weight", "price_usd", "mvrv"]

    with pytest.raises(ValueError, match="non-empty Polars interval string"):
        _series().resample("")
    with pytest.raises(ValueError, match="model must be one of"):
        _series().decompose(period=2, model="bad", series="price")
    with pytest.raises(ValueError, match="period must be an integer > 1"):
        _series().decompose(period=1, model="additive", series="price")

    bad = _series([100.0, 0.0, 110.0, 115.0, 120.0, 125.0])
    with pytest.raises(ValueError, match="strictly positive"):
        bad.decompose(period=2, model="multiplicative", series="price")

    good = _series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    dec = good.decompose(period=20, model="multiplicative", series="price")
    assert "seasonal" in dec.columns
    assert len(dec) == 6


def test_detrend_and_difference_error_and_branch_paths() -> None:
    ts = _series()
    with pytest.raises(ValueError, match="method must be one of"):
        ts.detrend(method="bad")
    with pytest.raises(ValueError, match="Unknown columns for detrend"):
        ts.detrend(columns=["missing"])

    diffed = ts.detrend(method="difference", columns=["price_usd"])
    assert "price_usd_detrended" in diffed.columns

    sparse_metric = ts.to_dataframe().with_columns(
        pl.Series("mvrv", [None, None, None, None, None, 1.0], dtype=pl.Float64)
    )
    detrended = WeightTimeSeries(metadata=ts.metadata, data=sparse_metric).detrend(
        method="linear",
        columns=["mvrv"],
    )
    assert (
        detrended["mvrv_detrended"].is_nan() | detrended["mvrv_detrended"].is_null()
    ).all()

    with pytest.raises(ValueError, match="Unknown columns for difference"):
        ts.difference(columns=["unknown_col"])
    with pytest.raises(ValueError, match="order must be >= 0"):
        ts.difference(order=-1)
    with pytest.raises(ValueError, match="seasonal_order must be >= 0"):
        ts.difference(order=0, seasonal_order=-1)
    with pytest.raises(ValueError, match="seasonal_period must be a positive integer"):
        ts.difference(order=0, seasonal_order=1, seasonal_period=0)

    seasonal = ts.difference(
        order=1,
        seasonal_order=1,
        seasonal_period=2,
        columns=["price_usd"],
    )
    assert "price_usd_diff" in seasonal.columns


def test_acf_pacf_spectral_and_stationarity_edge_paths() -> None:
    ts = _series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    acf = ts.acf_pacf(lags=[1, 100], series="returns")
    row_high = acf.filter(pl.col("lag") == 100).row(0, named=True)
    assert row_high["acf"] is None or (
        isinstance(row_high["acf"], float) and np.isnan(row_high["acf"])
    )
    assert row_high["pacf"] is None or (
        isinstance(row_high["pacf"], float) and np.isnan(row_high["pacf"])
    )

    ts_nan = _series([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    nan_prices = ts_nan.to_dataframe().with_columns(pl.lit(float("nan")).alias("price_usd"))
    empty_spec = WeightTimeSeries(metadata=ts_nan.metadata, data=nan_prices).spectral_density(
        series="returns"
    )
    assert empty_spec.is_empty()
    with pytest.raises(ValueError, match="method must be 'periodogram'"):
        ts.spectral_density(method="welch")

    assert WeightTimeSeries._stationarity_proxy(pl.Series("s", [1.0, 2.0]), acf_threshold=0.8)
    assert WeightTimeSeries._stationarity_proxy(pl.Series("s", [1.0, 1.0, 1.0]), acf_threshold=0.8)
    assert WeightTimeSeries._stationarity_proxy(
        pl.Series("s", [1.0, None, 1.0], dtype=pl.Float64),
        acf_threshold=0.8,
    )


def test_multiplicative_decompose_seasonal_mean_fallback_and_stationarity_nan_lag(monkeypatch) -> None:
    ts = _series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    nan_prices = ts.to_dataframe().with_columns(pl.lit(float("nan")).alias("price_usd"))
    ts_all_nan = WeightTimeSeries(metadata=ts.metadata, data=nan_prices)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dec = ts_all_nan.decompose(period=2, model="multiplicative", series="price")
    assert "seasonal" in dec.columns
    assert len(dec) == len(ts.data)
    assert caught == []

    monkeypatch.setattr(
        "stacksats.strategy_time_series_analysis._autocorr_pl",
        lambda s, lag: None,
    )
    assert WeightTimeSeries._stationarity_proxy(
        pl.Series("s", [1.0, 2.0, 3.0]),
        acf_threshold=0.8,
    )

    orig_isfinite = np.isfinite

    def _patched_isfinite(value):
        if isinstance(value, float) and value == 1.0:
            return False
        return orig_isfinite(value)

    monkeypatch.setattr("stacksats.strategy_time_series_analysis.np.isfinite", _patched_isfinite)
    dec_fallback = ts.decompose(period=2, model="multiplicative", series="price")
    assert "residual" in dec_fallback.columns


def test_calendar_cross_correlation_and_integration_order_edge_paths() -> None:
    ts = _series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])

    month = ts.seasonality_profile(freq="month", series="returns")
    assert month.height == 12
    assert month.filter(pl.col("count") == 0).height >= 1

    with pytest.raises(ValueError, match="freq must be one of"):
        ts.seasonality_profile(freq="year")
    with pytest.raises(ValueError, match="max_lag must be >= 0"):
        ts.cross_correlation("mvrv", max_lag=-1)

    empty_other = pl.Series("other", [None, None, None, None, None, None], dtype=pl.Float64)
    corr = ts.cross_correlation(empty_other, max_lag=1)
    assert corr["correlation"].null_count() == corr.height

    with pytest.raises(ValueError, match="max_order must be >= 0"):
        ts.integration_order(max_order=-1)
    with pytest.raises(ValueError, match="acf_threshold must be between 0 and 1"):
        ts.integration_order(acf_threshold=1.0)
    with pytest.raises(ValueError, match="Unknown columns for integration_order"):
        ts.integration_order(columns=["missing"])
