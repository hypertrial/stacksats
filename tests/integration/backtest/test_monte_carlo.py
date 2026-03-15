"""Monte Carlo robustness coverage for Polars-native backtests."""

from __future__ import annotations

import numpy as np
import pytest

from stacksats.model_development import compute_weights_fast, precompute_features
from tests.integration.backtest.polars_backtest_testkit import (
    dt_at,
    make_btc_df_from_prices,
)

pytestmark = pytest.mark.integration


def _prices_from_returns(returns: np.ndarray, start_price: float = 20000.0) -> np.ndarray:
    return start_price * np.exp(np.cumsum(returns))


def random_walk_prices(*, days: int = 365 * 6, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0004, 0.03, size=days)
    return _prices_from_returns(returns)


def fat_tail_prices(*, days: int = 365 * 6, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    returns = rng.standard_t(df=3, size=days) * 0.02
    return _prices_from_returns(returns)


def mean_reverting_prices(*, days: int = 365 * 6, seed: int = 13) -> np.ndarray:
    rng = np.random.default_rng(seed)
    prices = np.empty(days)
    prices[0] = 25000.0
    mean_price = 30000.0
    for idx in range(1, days):
        shock = rng.normal(0.0, 900.0)
        prices[idx] = max(1000.0, prices[idx - 1] + 0.03 * (mean_price - prices[idx - 1]) + shock)
    return prices


def regime_switching_prices(*, days: int = 365 * 6, seed: int = 17) -> np.ndarray:
    rng = np.random.default_rng(seed)
    prices = np.empty(days)
    prices[0] = 18000.0
    for idx in range(1, days):
        cycle = (idx // 180) % 3
        drift = [0.001, -0.0015, 0.0][cycle]
        vol = [0.02, 0.04, 0.015][cycle]
        prices[idx] = max(1000.0, prices[idx - 1] * np.exp(drift + rng.normal(0.0, vol)))
    return prices


def _weights_from_prices(prices: np.ndarray):
    btc_df = make_btc_df_from_prices(prices, start="2020-01-01")
    features_df = precompute_features(btc_df)
    return compute_weights_fast(features_df, dt_at("2024-01-01"), dt_at("2024-12-31"))


@pytest.mark.parametrize(
    "generator",
    [random_walk_prices, fat_tail_prices, mean_reverting_prices, regime_switching_prices],
)
def test_monte_carlo_paths_produce_valid_weights(generator):
    weights = _weights_from_prices(generator())
    values = weights["weight"].to_numpy()
    assert weights.height == 366
    assert np.isfinite(values).all()
    assert np.isclose(values.sum(), 1.0, rtol=1e-6)


class TestMonteCarloRandomWalk:
    def test_random_walk_paths_are_not_identical(self):
        first = random_walk_prices(seed=1)
        second = random_walk_prices(seed=2)
        assert not np.allclose(first, second)


class TestMonteCarloFatTails:
    def test_fat_tail_returns_have_higher_kurtosis_than_random_walk(self):
        random_walk_returns = np.diff(np.log(random_walk_prices(seed=3)))
        fat_tail_returns = np.diff(np.log(fat_tail_prices(seed=3)))
        random_kurtosis = float(np.mean((random_walk_returns - random_walk_returns.mean()) ** 4))
        fat_kurtosis = float(np.mean((fat_tail_returns - fat_tail_returns.mean()) ** 4))
        assert fat_kurtosis > random_kurtosis


class TestMonteCarloMeanReverting:
    def test_mean_reverting_path_still_allocates_full_budget(self):
        weights = _weights_from_prices(mean_reverting_prices())
        assert float(weights["weight"].sum()) == pytest.approx(1.0, abs=1e-10)


class TestMonteCarloRegimeSwitching:
    def test_regime_switching_path_differs_from_mean_reverting(self):
        regime = regime_switching_prices()
        mean_reverting = mean_reverting_prices()
        assert not np.allclose(regime, mean_reverting)


class TestMonteCarloRobustnessSummary:
    def test_multiple_runs_have_bounded_average_weight_std(self):
        stds = [
            float(_weights_from_prices(random_walk_prices(seed=seed))["weight"].std())
            for seed in range(5)
        ]
        assert np.isfinite(stds).all()
        assert float(np.mean(stds)) < 0.05


class TestWeightStabilityMonteCarlo:
    def test_neighboring_seeds_keep_weight_invariants(self):
        for seed in [21, 22]:
            weights = _weights_from_prices(random_walk_prices(seed=seed))["weight"].to_numpy()
            assert np.isfinite(weights).all()
            assert np.isclose(weights.sum(), 1.0, rtol=1e-6)
