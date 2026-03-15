"""Statistical validation smoke coverage for Polars-native backtests."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from stacksats.backtest import compute_weights_with_features
from stacksats.framework_contract import ALLOCATION_SPAN_DAYS
from stacksats.model_development import (
    compute_preference_scores,
    compute_weights_fast,
    precompute_features,
)
from stacksats.prelude import compute_cycle_spd
from tests.integration.backtest.polars_backtest_testkit import (
    dt_at,
    make_btc_df,
)

pytestmark = pytest.mark.integration


def _shared_strategy(features_df: pl.DataFrame):
    return lambda window_feat: compute_weights_with_features(window_feat, features_df=features_df)


@pytest.fixture(scope="module")
def stats_btc_df():
    return make_btc_df(start="2019-01-01", days=365 * 7, price_start=12000.0, price_step=20.0)


@pytest.fixture(scope="module")
def stats_features_df(stats_btc_df):
    return precompute_features(stats_btc_df)


class TestRandomizedBaseline:
    def test_shuffled_prices_change_spd_distribution(self, stats_btc_df, stats_features_df):
        shuffled_prices = np.array(stats_btc_df["price_usd"].to_list())
        rng = np.random.default_rng(7)
        rng.shuffle(shuffled_prices)
        shuffled_btc_df = stats_btc_df.with_columns(pl.Series("price_usd", shuffled_prices))
        shuffled_features = precompute_features(shuffled_btc_df)

        baseline_spd = compute_cycle_spd(
            stats_btc_df,
            _shared_strategy(stats_features_df),
            features_df=stats_features_df,
            start_date="2021-01-01",
            end_date="2024-12-31",
        )
        shuffled_spd = compute_cycle_spd(
            shuffled_btc_df,
            _shared_strategy(shuffled_features),
            features_df=shuffled_features,
            start_date="2021-01-01",
            end_date="2024-12-31",
        )

        assert not baseline_spd.is_empty()
        assert not shuffled_spd.is_empty()
        assert baseline_spd["dynamic_percentile"].mean() != pytest.approx(
            shuffled_spd["dynamic_percentile"].mean(),
            abs=1e-6,
        )


class TestOverfittingDetection:
    def test_signal_shuffle_preserves_finite_preference_scores(self, stats_features_df):
        start_date = dt_at("2024-01-01")
        end_date = dt_at("2024-06-30")

        shuffled = stats_features_df.with_columns(
            pl.Series("mvrv_zscore", list(reversed(stats_features_df["mvrv_zscore"].to_list())))
        )
        perturbed = compute_preference_scores(shuffled, start_date, end_date)

        assert perturbed.height > 0
        assert np.isfinite(perturbed["preference"].to_numpy()).all()


class TestFeatureValidity:
    def test_precomputed_features_remain_sorted_and_non_empty(self, stats_features_df):
        assert stats_features_df.height > 0
        assert stats_features_df["date"].is_sorted()
        assert stats_features_df["date"].null_count() == 0


class TestOutOfSample:
    def test_out_of_sample_backtest_is_non_empty(self, stats_btc_df):
        train_df = stats_btc_df.filter(pl.col("date") < dt_at("2024-01-01"))
        test_df = stats_btc_df.filter(pl.col("date") >= dt_at("2024-01-01"))
        del train_df
        test_features = precompute_features(test_df)

        spd = compute_cycle_spd(
            test_df,
            _shared_strategy(test_features),
            features_df=test_features,
            start_date="2024-01-01",
            end_date="2025-12-31",
        )
        assert not spd.is_empty()


class TestNumericalStability:
    def test_extreme_numeric_ranges_still_produce_finite_weights(self):
        prices = np.geomspace(1e2, 1e8, num=900)
        btc_df = make_btc_df(start="2022-01-01", days=900).with_columns(
            pl.Series("price_usd", prices),
            pl.Series("PriceUSD", prices),
            pl.Series("mvrv", np.linspace(0.4, 4.5, num=900)),
        )
        features_df = precompute_features(btc_df)
        weights = compute_weights_fast(features_df, dt_at("2024-01-01"), dt_at("2024-06-30"))
        assert np.isfinite(weights["weight"].to_numpy()).all()
        assert float(weights["weight"].sum()) == pytest.approx(1.0, abs=1e-10)


class TestSanityChecks:
    def test_explicit_window_helper_matches_direct_weight_call(self, stats_features_df):
        start_date = dt_at("2024-01-01")
        end_date = start_date + dt.timedelta(days=ALLOCATION_SPAN_DAYS - 1)
        window_feat = stats_features_df.filter(
            (pl.col("date") >= start_date) & (pl.col("date") <= end_date)
        )
        direct = compute_weights_fast(stats_features_df, start_date, end_date)
        via_backtest = compute_weights_with_features(window_feat, features_df=stats_features_df)
        np.testing.assert_allclose(
            direct["weight"].to_numpy(),
            via_backtest["weight"].to_numpy(),
            atol=1e-12,
        )
