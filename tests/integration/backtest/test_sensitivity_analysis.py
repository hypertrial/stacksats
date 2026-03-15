"""Sensitivity analysis for the Polars-native allocation pipeline."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import stacksats.model_development as md
from stacksats.model_development import (
    DYNAMIC_STRENGTH,
    compute_asymmetric_extreme_boost,
    compute_dynamic_multiplier,
    compute_weights_fast,
    precompute_features,
)
from tests.integration.backtest.polars_backtest_testkit import dt_at, make_btc_df

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def sensitivity_btc_df():
    return make_btc_df(start="2020-01-01", days=365 * 6, price_start=10000.0, price_step=22.0)


@pytest.fixture(scope="module")
def sensitivity_features_df(sensitivity_btc_df):
    return precompute_features(sensitivity_btc_df)


def weight_difference_metrics(weights1: pl.DataFrame, weights2: pl.DataFrame) -> dict[str, float]:
    joined = weights1.select(["date", pl.col("weight").alias("w1")]).join(
        weights2.select(["date", pl.col("weight").alias("w2")]),
        on="date",
        how="inner",
    )
    diff = joined["w1"].to_numpy() - joined["w2"].to_numpy()
    abs_diff = np.abs(diff)
    w1 = joined["w1"].to_numpy()
    w2 = joined["w2"].to_numpy()
    correlation = np.corrcoef(w1, w2)[0, 1] if joined.height > 1 else 1.0
    return {
        "max_abs_diff": float(abs_diff.max(initial=0.0)),
        "mean_abs_diff": float(abs_diff.mean() if abs_diff.size else 0.0),
        "rmse": float(np.sqrt(np.mean(diff**2)) if diff.size else 0.0),
        "max_relative_diff": float((abs_diff / np.maximum(w1, 1e-10)).max(initial=0.0)),
        "correlation": float(correlation),
    }


def compute_weights_with_dynamic_strength(
    features_df: pl.DataFrame,
    start_date,
    end_date,
    dynamic_strength: float,
) -> pl.DataFrame:
    original_strength = md.DYNAMIC_STRENGTH
    md.DYNAMIC_STRENGTH = dynamic_strength
    try:
        return compute_weights_fast(features_df, start_date, end_date)
    finally:
        md.DYNAMIC_STRENGTH = original_strength


class TestDynamicStrengthSensitivity:
    def test_small_dynamic_strength_perturbation(self, sensitivity_features_df):
        start_date = dt_at("2024-01-01")
        end_date = dt_at("2024-12-31")

        base_weights = compute_weights_fast(sensitivity_features_df, start_date, end_date)
        perturbed_weights = compute_weights_with_dynamic_strength(
            sensitivity_features_df,
            start_date,
            end_date,
            DYNAMIC_STRENGTH * 1.05,
        )

        metrics = weight_difference_metrics(base_weights, perturbed_weights)
        assert metrics["max_abs_diff"] < 0.15
        assert metrics["correlation"] > 0.95

    def test_dynamic_strength_range_produces_valid_weights(self, sensitivity_features_df):
        start_date = dt_at("2024-01-01")
        end_date = dt_at("2024-12-31")

        for strength in [1.0, 2.0, 3.0, 4.0, 5.0]:
            weights = compute_weights_with_dynamic_strength(
                sensitivity_features_df,
                start_date,
                end_date,
                strength,
            )
            values = weights["weight"].to_numpy()
            assert np.isclose(values.sum(), 1.0, rtol=1e-6)
            assert np.isfinite(values).all()
            assert (values >= -1e-10).all()


class TestZoneThresholdSensitivity:
    def test_zone_threshold_stability(self):
        mvrv_values = np.linspace(-3, 4, 100)
        base_zones = md.classify_mvrv_zone(mvrv_values)

        original = (
            md.MVRV_ZONE_DEEP_VALUE,
            md.MVRV_ZONE_VALUE,
            md.MVRV_ZONE_CAUTION,
            md.MVRV_ZONE_DANGER,
        )
        try:
            md.MVRV_ZONE_DEEP_VALUE *= 0.9
            md.MVRV_ZONE_VALUE *= 0.9
            md.MVRV_ZONE_CAUTION *= 1.1
            md.MVRV_ZONE_DANGER *= 1.1
            perturbed = md.classify_mvrv_zone(mvrv_values)
        finally:
            (
                md.MVRV_ZONE_DEEP_VALUE,
                md.MVRV_ZONE_VALUE,
                md.MVRV_ZONE_CAUTION,
                md.MVRV_ZONE_DANGER,
            ) = original

        assert float(np.mean(base_zones != perturbed)) < 0.3

    def test_extreme_boost_continuity(self):
        boosts = compute_asymmetric_extreme_boost(np.linspace(-4, 4, 1000))
        assert float(np.abs(np.diff(boosts)).max()) < 0.6


class TestMultiplierComponentSensitivity:
    def test_dynamic_multiplier_components_remain_positive(self):
        n = 366
        result = compute_dynamic_multiplier(
            np.random.uniform(-1, 1, n),
            np.random.uniform(-4, 4, n),
            np.random.uniform(-1, 1, n),
            np.random.uniform(0, 1, n),
            np.random.uniform(-1, 1, n),
            np.random.uniform(0, 1, n),
            np.random.uniform(0, 1, n),
        )
        assert len(result) == n
        assert np.isfinite(result).all()
        assert (result > 0).all()

    def test_feature_perturbation_keeps_weights_stable(self, sensitivity_btc_df):
        base_features = precompute_features(sensitivity_btc_df)
        perturbed_btc = sensitivity_btc_df.with_columns(
            (pl.col("mvrv") * 1.02).alias("mvrv")
        )
        perturbed_features = precompute_features(perturbed_btc)

        start_date = dt_at("2024-01-01")
        end_date = dt_at("2024-12-31")
        base_weights = compute_weights_fast(base_features, start_date, end_date)
        perturbed_weights = compute_weights_fast(perturbed_features, start_date, end_date)

        metrics = weight_difference_metrics(base_weights, perturbed_weights)
        assert metrics["rmse"] < 0.05
        assert metrics["correlation"] > 0.9


class TestSensitivityReport:
    def test_report_metrics_are_finite(self, sensitivity_features_df):
        start_date = dt_at("2024-01-01")
        end_date = dt_at("2024-12-31")

        base_weights = compute_weights_fast(sensitivity_features_df, start_date, end_date)
        stronger_weights = compute_weights_with_dynamic_strength(
            sensitivity_features_df,
            start_date,
            end_date,
            DYNAMIC_STRENGTH * 1.1,
        )
        metrics = weight_difference_metrics(base_weights, stronger_weights)
        assert all(np.isfinite(value) for value in metrics.values())
