"""Parity coverage between direct backtest weights and export weights."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from stacksats.export_weights import process_start_date_batch
from stacksats.framework_contract import ALLOCATION_SPAN_DAYS
from stacksats.model_development import compute_window_weights, precompute_features
from tests.integration.backtest.polars_backtest_testkit import (
    dt_at,
    make_btc_df,
    normalize_weight_frame,
    weight_lookup,
)

pytestmark = pytest.mark.integration

PRICE_COL = "price_usd"


@pytest.fixture(scope="module")
def parity_btc_df():
    return make_btc_df(start="2020-01-01", days=2558, price_start=15000.0, price_step=30.0)


@pytest.fixture(scope="module")
def parity_features_df(parity_btc_df):
    return precompute_features(parity_btc_df)


def _window_end(start_date: dt.datetime) -> dt.datetime:
    return start_date + dt.timedelta(days=ALLOCATION_SPAN_DAYS - 1)


def _backtest_weights(
    features_df: pl.DataFrame,
    *,
    start_date: dt.datetime,
    current_date: dt.datetime,
) -> pl.DataFrame:
    return normalize_weight_frame(
        compute_window_weights(
            features_df=features_df,
            start_date=start_date,
            end_date=_window_end(start_date),
            current_date=current_date,
        )
    )


def _export_weights(
    features_df: pl.DataFrame,
    btc_df: pl.DataFrame,
    *,
    start_date: dt.datetime,
    current_date: dt.datetime,
) -> pl.DataFrame:
    return normalize_weight_frame(
        process_start_date_batch(
            start_date,
            [_window_end(start_date)],
            features_df,
            btc_df,
            current_date,
            PRICE_COL,
        ).select(["date", "weight"])
    )


class TestWeightComputationParity:
    def test_export_matches_backtest_after_window_end(self, parity_btc_df, parity_features_df):
        start_date = dt_at("2021-01-01")
        current_date = dt_at("2022-06-01")

        expected = _backtest_weights(
            parity_features_df,
            start_date=start_date,
            current_date=current_date,
        )
        actual = _export_weights(
            parity_features_df,
            parity_btc_df,
            start_date=start_date,
            current_date=current_date,
        )

        np.testing.assert_allclose(actual["weight"].to_numpy(), expected["weight"].to_numpy(), atol=1e-12)

    def test_export_matches_backtest_mid_window(self, parity_btc_df, parity_features_df):
        start_date = dt_at("2025-06-01")
        current_date = dt_at("2025-12-15")

        expected = _backtest_weights(
            parity_features_df,
            start_date=start_date,
            current_date=current_date,
        )
        actual = _export_weights(
            parity_features_df,
            parity_btc_df,
            start_date=start_date,
            current_date=current_date,
        )

        np.testing.assert_allclose(actual["weight"].to_numpy(), expected["weight"].to_numpy(), atol=1e-12)

    def test_export_matches_backtest_before_window_start(self, parity_btc_df, parity_features_df):
        start_date = dt_at("2026-01-01")
        current_date = dt_at("2025-06-01")

        expected = _backtest_weights(
            parity_features_df,
            start_date=start_date,
            current_date=current_date,
        )
        actual = _export_weights(
            parity_features_df,
            parity_btc_df,
            start_date=start_date,
            current_date=current_date,
        )

        np.testing.assert_allclose(actual["weight"].to_numpy(), expected["weight"].to_numpy(), atol=1e-12)


class TestPastWeightImmutabilityParity:
    def test_past_prefix_is_immutable_across_current_dates(self, parity_btc_df, parity_features_df):
        start_date = dt_at("2021-01-01")
        current_date_1 = dt_at("2021-06-15")
        current_date_2 = dt_at("2021-09-15")

        weights_1 = _export_weights(
            parity_features_df,
            parity_btc_df,
            start_date=start_date,
            current_date=current_date_1,
        )
        weights_2 = _export_weights(
            parity_features_df,
            parity_btc_df,
            start_date=start_date,
            current_date=current_date_2,
        )

        lookup_1 = weight_lookup(weights_1)
        lookup_2 = weight_lookup(weights_2)
        past_dates = [date for date in lookup_1 if date <= current_date_1]

        for date in past_dates:
            assert lookup_1[date] == pytest.approx(lookup_2[date], abs=1e-12)

    def test_day_by_day_progression_only_changes_future_weights(self, parity_btc_df, parity_features_df):
        start_date = dt_at("2025-01-01")
        for day_offset in range(1, 7):
            current_date = start_date + dt.timedelta(days=day_offset)
            prev_current = current_date - dt.timedelta(days=1)

            current_weights = weight_lookup(
                _export_weights(
                    parity_features_df,
                    parity_btc_df,
                    start_date=start_date,
                    current_date=current_date,
                )
            )
            previous_weights = weight_lookup(
                _export_weights(
                    parity_features_df,
                    parity_btc_df,
                    start_date=start_date,
                    current_date=prev_current,
                )
            )

            for date, weight in previous_weights.items():
                if date <= prev_current:
                    assert current_weights[date] == pytest.approx(weight, abs=1e-12)


class TestWeightSumParity:
    @pytest.mark.parametrize(
        ("start_date", "current_date"),
        [
            (dt_at("2021-01-01"), dt_at("2022-01-01")),
            (dt_at("2025-01-01"), dt_at("2025-06-15")),
            (dt_at("2026-01-01"), dt_at("2025-01-01")),
        ],
    )
    def test_export_and_backtest_both_sum_to_one(
        self,
        parity_btc_df,
        parity_features_df,
        start_date,
        current_date,
    ):
        expected = _backtest_weights(
            parity_features_df,
            start_date=start_date,
            current_date=current_date,
        )
        actual = _export_weights(
            parity_features_df,
            parity_btc_df,
            start_date=start_date,
            current_date=current_date,
        )

        assert float(expected["weight"].sum()) == pytest.approx(1.0, abs=1e-10)
        assert float(actual["weight"].sum()) == pytest.approx(1.0, abs=1e-10)


class TestFutureWeightUniformityParity:
    def test_pre_start_current_date_produces_uniform_weights(self, parity_btc_df, parity_features_df):
        start_date = dt_at("2025-01-01")
        current_date = dt_at("2024-06-15")
        weights = _export_weights(
            parity_features_df,
            parity_btc_df,
            start_date=start_date,
            current_date=current_date,
        )

        expected = np.full(weights.height, 1.0 / weights.height)
        np.testing.assert_allclose(weights["weight"].to_numpy(), expected, atol=1e-12)


class TestEdgeCasesParity:
    def test_same_day_current_date_locks_first_weight_only(self, parity_btc_df, parity_features_df):
        start_date = dt_at("2021-06-15")
        current_date = start_date

        weights = _export_weights(
            parity_features_df,
            parity_btc_df,
            start_date=start_date,
            current_date=current_date,
        )
        assert weights.height == ALLOCATION_SPAN_DAYS
        assert float(weights["weight"].sum()) == pytest.approx(1.0, abs=1e-10)

    def test_leap_year_window_has_expected_length(self, parity_btc_df, parity_features_df):
        start_date = dt_at("2024-02-28")
        current_date = dt_at("2024-03-01")
        weights = _export_weights(
            parity_features_df,
            parity_btc_df,
            start_date=start_date,
            current_date=current_date,
        )
        assert weights.height == ALLOCATION_SPAN_DAYS

    def test_direct_compute_and_export_match_for_multiple_windows(self, parity_btc_df, parity_features_df):
        for start_date in [
            dt_at("2021-01-01"),
            dt_at("2025-06-01"),
            dt_at("2026-01-01"),
        ]:
            current_date = min(_window_end(start_date), dt_at("2025-12-15"))
            expected = _backtest_weights(
                parity_features_df,
                start_date=start_date,
                current_date=current_date,
            )
            actual = _export_weights(
                parity_features_df,
                parity_btc_df,
                start_date=start_date,
                current_date=current_date,
            )
            np.testing.assert_allclose(
                actual["weight"].to_numpy(),
                expected["weight"].to_numpy(),
                atol=1e-12,
            )
