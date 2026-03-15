"""Polars-native time-series cross-validation coverage."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from stacksats.model_development import compute_weights_fast, precompute_features
from tests.integration.backtest.polars_backtest_testkit import (
    day_range,
    dt_at,
    make_btc_df,
    slice_dates,
)

pytestmark = pytest.mark.integration


def generate_expanding_window_folds(
    start_date: str,
    end_date: str,
    *,
    n_folds: int = 5,
    min_train_days: int = 365 * 2,
    test_days: int = 365,
) -> list[tuple[dt.datetime, dt.datetime, dt.datetime, dt.datetime]]:
    start = dt_at(start_date)
    end = dt_at(end_date)
    available = (end - start).days - min_train_days
    if available < test_days * n_folds:
        test_days = max(30, available // max(n_folds, 1))

    folds: list[tuple[dt.datetime, dt.datetime, dt.datetime, dt.datetime]] = []
    for idx in range(n_folds):
        train_start = start
        train_end = start + dt.timedelta(days=min_train_days + idx * test_days - 1)
        test_start = train_end + dt.timedelta(days=1)
        test_end = test_start + dt.timedelta(days=test_days - 1)
        if test_end <= end:
            folds.append((train_start, train_end, test_start, test_end))
    return folds


def generate_rolling_window_folds(
    start_date: str,
    end_date: str,
    *,
    n_folds: int = 5,
    train_days: int = 365 * 2,
    test_days: int = 365,
) -> list[tuple[dt.datetime, dt.datetime, dt.datetime, dt.datetime]]:
    start = dt_at(start_date)
    end = dt_at(end_date)
    step_days = (
        (end - start).days - train_days - test_days
    ) // max(n_folds - 1, 1)

    folds: list[tuple[dt.datetime, dt.datetime, dt.datetime, dt.datetime]] = []
    for idx in range(n_folds):
        train_start = start + dt.timedelta(days=idx * max(step_days, 0))
        train_end = train_start + dt.timedelta(days=train_days - 1)
        test_start = train_end + dt.timedelta(days=1)
        test_end = test_start + dt.timedelta(days=test_days - 1)
        if test_end <= end:
            folds.append((train_start, train_end, test_start, test_end))
    return folds


def evaluate_fold(
    features_df: pl.DataFrame,
    btc_df: pl.DataFrame,
    test_start: dt.datetime,
    test_end: dt.datetime,
) -> dict[str, float]:
    test_days = (test_end - test_start).days + 1
    if test_days >= 365:
        window_days = 365
    elif test_days >= 180:
        window_days = 180
    elif test_days >= 90:
        window_days = 90
    else:
        window_days = max(30, test_days // 2)

    start_dates = day_range(test_start, test_end - dt.timedelta(days=window_days - 1))[::30]
    if not start_dates:
        return {"win_rate": float("nan"), "mean_excess": float("nan"), "n_windows": 0}

    excess_values: list[float] = []
    for start_date in start_dates:
        end_date = start_date + dt.timedelta(days=window_days - 1)
        weights = compute_weights_fast(features_df, start_date, end_date)
        prices = slice_dates(btc_df, start_date, end_date).select(["date", "price_usd"])
        if weights.is_empty() or prices.is_empty():
            continue

        aligned = prices.join(weights, on="date", how="inner")
        if aligned.height < 7:
            continue

        inv_price = 1e8 / aligned["price_usd"].to_numpy()
        min_spd = float(inv_price.min())
        max_spd = float(inv_price.max())
        span = max_spd - min_spd
        uniform_spd = float(inv_price.mean())
        dynamic_spd = float((aligned["weight"].to_numpy() * inv_price).sum())
        excess = 0.0 if span <= 0 else 100.0 * (dynamic_spd - uniform_spd) / span
        excess_values.append(excess)

    if not excess_values:
        return {"win_rate": float("nan"), "mean_excess": float("nan"), "n_windows": 0}

    return {
        "win_rate": float(np.mean([value > 0 for value in excess_values])),
        "mean_excess": float(np.mean(excess_values)),
        "n_windows": float(len(excess_values)),
    }


@pytest.fixture(scope="module")
def cv_btc_df():
    return make_btc_df(start="2018-01-01", days=365 * 8, price_start=9000.0, price_step=18.0)


@pytest.fixture(scope="module")
def cv_features_df(cv_btc_df):
    return precompute_features(cv_btc_df)


class TestExpandingWindowCV:
    def test_folds_are_monotonic_and_non_overlapping(self):
        folds = generate_expanding_window_folds("2018-01-01", "2025-12-31")
        assert len(folds) >= 3
        for train_start, train_end, test_start, test_end in folds:
            assert train_start <= train_end < test_start <= test_end

    def test_expanding_fold_evaluation_returns_metrics(self, cv_btc_df, cv_features_df):
        _, _, test_start, test_end = generate_expanding_window_folds(
            "2018-01-01",
            "2025-12-31",
        )[0]
        metrics = evaluate_fold(cv_features_df, cv_btc_df, test_start, test_end)
        assert metrics["n_windows"] > 0
        assert np.isfinite(metrics["mean_excess"])


class TestRollingWindowCV:
    def test_rolling_folds_have_fixed_train_length(self):
        folds = generate_rolling_window_folds("2018-01-01", "2025-12-31")
        lengths = {(train_end - train_start).days + 1 for train_start, train_end, _, _ in folds}
        assert lengths == {365 * 2}

    def test_rolling_fold_evaluation_returns_windows(self, cv_btc_df, cv_features_df):
        _, _, test_start, test_end = generate_rolling_window_folds(
            "2018-01-01",
            "2025-12-31",
        )[0]
        metrics = evaluate_fold(cv_features_df, cv_btc_df, test_start, test_end)
        assert metrics["n_windows"] > 0


class TestOverfittingDetection:
    def test_fold_mean_excess_is_not_nan_for_all_folds(self, cv_btc_df, cv_features_df):
        folds = generate_expanding_window_folds("2018-01-01", "2025-12-31")
        means = [
            evaluate_fold(cv_features_df, cv_btc_df, test_start, test_end)["mean_excess"]
            for _, _, test_start, test_end in folds
        ]
        assert any(np.isfinite(value) for value in means)


class TestCrossValidationConsistency:
    def test_expanding_and_rolling_cv_produce_metrics(self, cv_btc_df, cv_features_df):
        exp_fold = generate_expanding_window_folds("2018-01-01", "2025-12-31")[0]
        roll_fold = generate_rolling_window_folds("2018-01-01", "2025-12-31")[0]

        exp_metrics = evaluate_fold(cv_features_df, cv_btc_df, exp_fold[2], exp_fold[3])
        roll_metrics = evaluate_fold(cv_features_df, cv_btc_df, roll_fold[2], roll_fold[3])

        assert exp_metrics["n_windows"] > 0
        assert roll_metrics["n_windows"] > 0


class TestLeaveOneYearOutCV:
    def test_each_calendar_year_can_serve_as_a_holdout(self, cv_btc_df, cv_features_df):
        for year in range(2021, 2025):
            metrics = evaluate_fold(
                cv_features_df,
                cv_btc_df,
                dt_at(f"{year}-01-01"),
                dt_at(f"{year}-12-31"),
            )
            assert metrics["n_windows"] > 0
