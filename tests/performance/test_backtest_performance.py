"""Polars-native performance coverage for backtest helpers."""

from __future__ import annotations

import datetime as dt
import time

import pytest

from stacksats.backtest import compute_weights_with_features
from stacksats.framework_contract import ALLOCATION_SPAN_DAYS
from stacksats.model_development import compute_weights_fast, precompute_features
from stacksats.prelude import compute_cycle_spd
from tests.integration.backtest.polars_backtest_testkit import (
    dt_at,
    frame_has_date,
    make_btc_df,
    slice_dates,
)


def _shared_strategy(features_df):
    return lambda window_feat: compute_weights_with_features(
        window_feat,
        features_df=features_df,
    )


@pytest.mark.performance
class TestFeaturePrecomputationPerformance:
    def test_feature_precompute_time(self, sample_btc_df):
        start = time.time()
        precompute_features(sample_btc_df)
        elapsed = time.time() - start
        assert elapsed < 5.0

    def test_feature_precompute_scales_reasonably(self, sample_btc_df):
        half_df = sample_btc_df.head(sample_btc_df.height // 2)

        start_half = time.time()
        precompute_features(half_df)
        time_half = time.time() - start_half

        start_full = time.time()
        precompute_features(sample_btc_df)
        time_full = time.time() - start_full

        assert time_full < time_half * 4


@pytest.mark.performance
class TestWeightComputationPerformance:
    def test_single_window_weight_computation(self, sample_features_df):
        start_date = dt_at("2024-01-01")
        end_date = dt_at("2024-12-31")

        if not frame_has_date(sample_features_df, start_date):
            pytest.skip("Start date not in features date column")
        if not frame_has_date(sample_features_df, end_date):
            pytest.skip("End date not in features date column")

        compute_weights_fast(sample_features_df, start_date, end_date)

        start = time.time()
        compute_weights_fast(sample_features_df, start_date, end_date)
        elapsed = time.time() - start
        assert elapsed < 0.1

    def test_multiple_windows_weight_computation(self, sample_features_df):
        base_date = dt_at("2024-01-01")
        max_date = sample_features_df["date"].max()

        if not frame_has_date(sample_features_df, base_date):
            pytest.skip("Base date not in features date column")

        start = time.time()
        for offset in range(100):
            start_date = base_date + dt.timedelta(days=offset)
            end_date = start_date + dt.timedelta(days=30)
            if end_date > max_date:
                break
            compute_weights_fast(sample_features_df, start_date, end_date)
        elapsed = time.time() - start

        assert elapsed < 5.0

    def test_compute_weights_with_features_performance(self, sample_features_df):
        start_date = dt_at("2024-01-01")
        end_date = start_date + dt.timedelta(days=ALLOCATION_SPAN_DAYS - 1)

        if not frame_has_date(sample_features_df, start_date):
            pytest.skip("Start date not in features date column")
        if not frame_has_date(sample_features_df, end_date):
            pytest.skip("End date not in features date column")

        window_feat = slice_dates(sample_features_df, start_date, end_date)
        compute_weights_with_features(window_feat, features_df=sample_features_df)

        start = time.time()
        compute_weights_with_features(window_feat, features_df=sample_features_df)
        elapsed = time.time() - start
        assert elapsed < 0.1


@pytest.mark.performance
class TestFullBacktestPerformance:
    def test_backtest_completes_in_reasonable_time(self, sample_btc_df, sample_features_df):
        start = time.time()
        spd_table = compute_cycle_spd(
            sample_btc_df,
            _shared_strategy(sample_features_df),
            features_df=sample_features_df,
        )
        elapsed = time.time() - start

        per_window = elapsed / max(spd_table.height, 1)
        assert elapsed < 60.0
        assert per_window < 0.1


@pytest.mark.performance
class TestMemoryEfficiency:
    def test_weight_computation_returns_appropriate_size(self, sample_features_df):
        start_date = dt_at("2024-01-01")
        end_date = dt_at("2024-12-31")

        if not frame_has_date(sample_features_df, start_date):
            pytest.skip("Start date not in features date column")
        if not frame_has_date(sample_features_df, end_date):
            pytest.skip("End date not in features date column")

        weights = compute_weights_fast(sample_features_df, start_date, end_date)
        expected_len = slice_dates(sample_features_df, start_date, end_date).height
        assert weights.height == expected_len
        assert weights.estimated_size() < 1_000_000

    def test_features_df_size_is_reasonable(self, sample_btc_df):
        features_df = precompute_features(sample_btc_df)
        mem_mb = features_df.estimated_size() / (1024 * 1024)
        assert mem_mb < 100


@pytest.mark.performance
class TestScalability:
    def test_precompute_features_scales_with_longer_history(self):
        small_btc = make_btc_df(start="2022-01-01", days=365)
        large_btc = make_btc_df(start="2020-01-01", days=365 * 4)

        start_small = time.time()
        precompute_features(small_btc)
        small_elapsed = time.time() - start_small

        start_large = time.time()
        precompute_features(large_btc)
        large_elapsed = time.time() - start_large

        assert large_elapsed < max(small_elapsed * 8, 2.0)
