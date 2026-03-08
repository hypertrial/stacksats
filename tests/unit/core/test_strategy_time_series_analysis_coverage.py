from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from stacksats.strategy_time_series import StrategySeriesMetadata, StrategyTimeSeries


def _series(prices: list[float] | None = None) -> StrategyTimeSeries:
    price_vals = prices or [100.0, 95.0, 90.0, 85.0, 80.0, 82.0]
    dates = pd.date_range("2024-01-01", periods=len(price_vals), freq="D")
    data = pd.DataFrame(
        {
            "date": dates,
            "weight": [1.0 / len(price_vals)] * len(price_vals),
            "price_usd": price_vals,
            "mvrv": np.arange(len(price_vals), dtype=float),
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
    return StrategyTimeSeries(metadata=meta, data=data)


def test_eda_series_variants_and_errors() -> None:
    ts = _series([100.0, 0.0, -1.0, 3.0, 4.0, 5.0])
    log_returns = ts._eda_value_series("log_returns")
    assert len(log_returns) == len(ts.data)
    weight = ts._eda_value_series("weight")
    assert np.isfinite(weight.to_numpy(dtype=float)).all()
    with pytest.raises(ValueError, match="series must be one of"):
        ts._eda_value_series("bad-series")


def test_drawdown_branches_no_valid_and_unrecovered() -> None:
    ts = _series()
    empty_df = ts.to_dataframe()
    empty_df["price_usd"] = np.nan
    empty = StrategyTimeSeries(metadata=ts.metadata, data=empty_df).drawdown_table()
    assert empty.empty

    unrecovered = _series([100.0, 90.0, 80.0, 70.0, 60.0, 50.0]).drawdown_table(top_n=3)
    assert not unrecovered.empty
    assert bool((unrecovered["recovered"] == False).any())  # noqa: E712


def test_resolve_series_like_and_cross_correlation_paths() -> None:
    ts = _series()
    short_other = pd.Series([1.0, 2.0])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        corr = ts.cross_correlation(short_other, max_lag=2, series="returns")
        assert not corr.empty
        corr_col = ts.cross_correlation("mvrv", max_lag=1, series="returns")
        assert not corr_col.empty


def test_pacf_edge_paths() -> None:
    values = pd.Series([1.0, 1.0, 1.0, 1.0])
    assert StrategyTimeSeries._pacf_at_lag(values, 0) is None
    assert StrategyTimeSeries._pacf_at_lag(values.iloc[:2], 2) is None
    # Zero-variance residual branch.
    assert StrategyTimeSeries._pacf_at_lag(values, 2) is None


def test_resample_and_decompose_error_paths() -> None:
    ts = _series()
    non_numeric = ts.to_dataframe()
    non_numeric["weight"] = non_numeric["weight"].astype(str)
    non_numeric["price_usd"] = non_numeric["price_usd"].astype(str)
    non_numeric["mvrv"] = non_numeric["mvrv"].astype(str)
    object.__setattr__(ts, "_data", non_numeric)
    out = ts.resample("D")
    assert list(out.columns) == ["date"]

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

    diffed = ts.detrend(method="difference", columns=["price_usd"])
    assert "price_usd_detrended" in diffed.columns

    ts_single = _series()
    sparse_metric = ts_single.to_dataframe()
    sparse_metric["mvrv"] = [np.nan, np.nan, np.nan, np.nan, np.nan, 1.0]
    detrended = StrategyTimeSeries(metadata=ts_single.metadata, data=sparse_metric).detrend(
        method="linear", columns=["mvrv"]
    )
    assert detrended["mvrv_detrended"].isna().all()

    with pytest.raises(ValueError, match="Unknown columns for difference"):
        ts.difference(columns=["unknown_col"])

    seasonal = ts.difference(order=1, seasonal_order=1, seasonal_period=2, columns=["price_usd"])
    assert "price_usd_diff" in seasonal.columns


def test_acf_pacf_spectral_and_stationarity_edge_paths() -> None:
    ts = _series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    acf = ts.acf_pacf(lags=[1, 100], series="returns")
    row_high = acf.loc[acf["lag"] == 100].iloc[0]
    assert pd.isna(row_high["acf"])
    assert pd.isna(row_high["pacf"])

    ts_nan = _series([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    nan_prices = ts_nan.to_dataframe()
    nan_prices["price_usd"] = np.nan
    empty_spec = StrategyTimeSeries(metadata=ts_nan.metadata, data=nan_prices).spectral_density(
        series="returns"
    )
    assert empty_spec.empty

    assert StrategyTimeSeries._stationarity_proxy(pd.Series([1.0, 2.0]), acf_threshold=0.8)
    assert StrategyTimeSeries._stationarity_proxy(pd.Series([1.0, 1.0, 1.0]), acf_threshold=0.8)
    assert StrategyTimeSeries._stationarity_proxy(pd.Series([1.0, np.nan, 1.0]), acf_threshold=0.8)


def test_multiplicative_decompose_seasonal_mean_fallback_and_stationarity_nan_lag(monkeypatch) -> None:
    ts = _series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    nan_prices = ts.to_dataframe()
    nan_prices["price_usd"] = np.nan
    ts_all_nan = StrategyTimeSeries(metadata=ts.metadata, data=nan_prices)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        dec = ts_all_nan.decompose(period=2, model="multiplicative", series="price")
    assert "seasonal" in dec.columns
    assert len(dec) == len(ts.data)

    monkeypatch.setattr(pd.Series, "autocorr", lambda self, lag=1: np.nan)
    assert StrategyTimeSeries._stationarity_proxy(pd.Series([1.0, 2.0, 3.0]), acf_threshold=0.8)
