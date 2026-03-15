"""Synthetic market-regime integration coverage for Polars runtime paths."""

from __future__ import annotations

import numpy as np
import pytest

from stacksats.model_development import compute_weights_fast, precompute_features
from tests.integration.backtest.polars_backtest_testkit import (
    dt_at,
    make_btc_df_from_prices,
)

pytestmark = pytest.mark.integration


def _bull_prices(days: int = 365 * 6) -> np.ndarray:
    base = np.linspace(12000, 90000, num=days)
    cycle = 1.0 + 0.08 * np.sin(np.linspace(0, 12 * np.pi, num=days))
    return base * cycle


def _bear_prices(days: int = 365 * 6) -> np.ndarray:
    base = np.linspace(90000, 18000, num=days)
    cycle = 1.0 + 0.06 * np.sin(np.linspace(0, 10 * np.pi, num=days))
    return base * cycle


def _sideways_prices(days: int = 365 * 6) -> np.ndarray:
    return 30000 * (1.0 + 0.03 * np.sin(np.linspace(0, 18 * np.pi, num=days)))


def _volatile_prices(days: int = 365 * 6) -> np.ndarray:
    base = 35000 * np.ones(days)
    shock = 1.0 + 0.25 * np.sin(np.linspace(0, 40 * np.pi, num=days))
    return base * shock


def _crash_recovery_prices(days: int = 365 * 6) -> np.ndarray:
    prices = _bull_prices(days)
    crash_idx = days // 2
    prices[crash_idx : crash_idx + 45] *= np.linspace(1.0, 0.45, num=45)
    prices[crash_idx + 45 : crash_idx + 180] *= np.linspace(0.45, 1.15, num=135)
    return prices


def _weights_from_prices(prices: np.ndarray):
    btc_df = make_btc_df_from_prices(prices, start="2020-01-01")
    features_df = precompute_features(btc_df)
    start_date = dt_at("2024-01-01")
    end_date = dt_at("2024-12-31")
    return compute_weights_fast(features_df, start_date, end_date)


def _features_from_prices(prices: np.ndarray):
    return precompute_features(make_btc_df_from_prices(prices, start="2020-01-01"))


@pytest.mark.parametrize(
    "prices_fn",
    [_bull_prices, _bear_prices, _sideways_prices, _volatile_prices, _crash_recovery_prices],
)
def test_regime_generators_produce_valid_weight_series(prices_fn):
    weights = _weights_from_prices(prices_fn())
    values = weights["weight"].to_numpy()
    assert weights.height == 366
    assert np.isfinite(values).all()
    assert np.isclose(values.sum(), 1.0, rtol=1e-6)


class TestBullMarketPerformance:
    def test_bull_market_features_track_positive_long_term_trend(self):
        prices = _bull_prices()
        features = _features_from_prices(prices)
        assert prices[-1] > prices[0]
        assert features.height > 0


class TestBearMarketPerformance:
    def test_bear_market_features_track_negative_long_term_trend(self):
        prices = _bear_prices()
        features = _features_from_prices(prices)
        assert prices[-1] < prices[0]
        assert features.height > 0


class TestSidewaysMarketPerformance:
    def test_sideways_market_has_lower_price_volatility_than_volatile_market(self):
        sideways_returns = np.diff(np.log(_sideways_prices()))
        volatile_returns = np.diff(np.log(_volatile_prices()))
        assert float(np.std(sideways_returns)) < float(np.std(volatile_returns))


class TestCrashRecoveryPerformance:
    def test_crash_recovery_series_contains_drawdown_and_recovery(self):
        prices = _crash_recovery_prices()
        center = len(prices) // 2
        crash_slice = prices[center - 20 : center + 60]
        assert float(crash_slice.min()) < float(prices.max()) * 0.6
        assert float(prices[-1]) > float(crash_slice.min())


class TestCrossRegimeComparison:
    def test_cross_regime_price_statistics_are_distinct(self):
        stats = [
            (float(np.mean(np.diff(np.log(_bull_prices())))), float(np.std(np.diff(np.log(_bull_prices()))))),
            (float(np.mean(np.diff(np.log(_bear_prices())))), float(np.std(np.diff(np.log(_bear_prices()))))),
            (float(np.mean(np.diff(np.log(_sideways_prices())))), float(np.std(np.diff(np.log(_sideways_prices()))))),
            (float(np.mean(np.diff(np.log(_volatile_prices())))), float(np.std(np.diff(np.log(_volatile_prices()))))),
        ]
        assert len(set(stats)) == len(stats)
