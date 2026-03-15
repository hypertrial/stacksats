"""Daily simulation coverage for the Polars-only allocation kernel."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from stacksats.framework_contract import ALLOCATION_SPAN_DAYS
from stacksats.model_development import compute_window_weights, precompute_features
from tests.integration.backtest.polars_backtest_testkit import (
    dt_at,
    make_btc_df,
    normalize_weight_frame,
    weight_lookup,
)

pytestmark = pytest.mark.integration


class StrategyWindowSimulator:
    """Minimal daily simulator over the sealed framework allocation kernel."""

    def __init__(self, features_df, start_date: str):
        self.features_df = features_df
        self.start_date = dt_at(start_date)
        self.end_date = self.start_date + dt.timedelta(days=ALLOCATION_SPAN_DAYS - 1)

    def weights_for(self, current_date: str | dt.datetime):
        return normalize_weight_frame(
            compute_window_weights(
                self.features_df,
                self.start_date,
                self.end_date,
                dt_at(current_date),
            )
        )


@pytest.fixture(scope="module")
def simulation_features_df():
    return precompute_features(make_btc_df(start="2020-01-01", days=2600, price_start=20000.0, price_step=18.0))


class TestScenarioAInitialPopulation:
    def test_pre_start_weights_are_uniform(self, simulation_features_df):
        simulator = StrategyWindowSimulator(simulation_features_df, "2025-01-01")
        weights = simulator.weights_for("2024-12-01")
        expected = np.full(weights.height, 1.0 / weights.height)
        np.testing.assert_allclose(weights["weight"].to_numpy(), expected, atol=1e-12)

    def test_start_day_population_is_complete(self, simulation_features_df):
        simulator = StrategyWindowSimulator(simulation_features_df, "2025-01-01")
        weights = simulator.weights_for("2025-01-01")
        assert weights.height == ALLOCATION_SPAN_DAYS
        assert float(weights["weight"].sum()) == pytest.approx(1.0, abs=1e-10)


class TestScenarioBDailyUpdates:
    def test_past_weights_lock_after_each_day(self, simulation_features_df):
        simulator = StrategyWindowSimulator(simulation_features_df, "2025-01-01")
        day_one = weight_lookup(simulator.weights_for("2025-01-01"))
        day_two = weight_lookup(simulator.weights_for("2025-01-02"))
        for date, weight in day_one.items():
            if date <= dt_at("2025-01-01"):
                assert day_two[date] == pytest.approx(weight, abs=1e-12)

    def test_two_consecutive_days_only_change_future_segment(self, simulation_features_df):
        simulator = StrategyWindowSimulator(simulation_features_df, "2025-01-01")
        prev = weight_lookup(simulator.weights_for("2025-02-10"))
        curr = weight_lookup(simulator.weights_for("2025-02-11"))
        cutoff = dt_at("2025-02-10")
        for date, weight in prev.items():
            if date <= cutoff:
                assert curr[date] == pytest.approx(weight, abs=1e-12)


class TestScenarioCMultiDayProgression:
    def test_prefix_remains_stable_over_weeklong_progression(self, simulation_features_df):
        simulator = StrategyWindowSimulator(simulation_features_df, "2025-01-01")
        for offset in range(1, 8):
            earlier = dt_at("2025-03-01") + dt.timedelta(days=offset - 1)
            later = earlier + dt.timedelta(days=1)
            earlier_weights = weight_lookup(simulator.weights_for(earlier))
            later_weights = weight_lookup(simulator.weights_for(later))
            for date, weight in earlier_weights.items():
                if date <= earlier:
                    assert later_weights[date] == pytest.approx(weight, abs=1e-12)

    def test_full_window_completion_preserves_sum_and_shape(self, simulation_features_df):
        simulator = StrategyWindowSimulator(simulation_features_df, "2025-01-01")
        weights = simulator.weights_for(simulator.end_date)
        assert weights.height == ALLOCATION_SPAN_DAYS
        assert float(weights["weight"].sum()) == pytest.approx(1.0, abs=1e-10)


class TestScenarioDEdgeCases:
    def test_current_after_window_end_keeps_same_final_weights(self, simulation_features_df):
        simulator = StrategyWindowSimulator(simulation_features_df, "2025-01-01")
        on_end = simulator.weights_for(simulator.end_date)
        after_end = simulator.weights_for(simulator.end_date + dt.timedelta(days=10))
        np.testing.assert_allclose(on_end["weight"].to_numpy(), after_end["weight"].to_numpy(), atol=1e-12)

    def test_leap_year_window_is_supported(self, simulation_features_df):
        simulator = StrategyWindowSimulator(simulation_features_df, "2024-02-28")
        weights = simulator.weights_for("2024-03-10")
        assert weights.height == ALLOCATION_SPAN_DAYS
        assert float(weights["weight"].sum()) == pytest.approx(1.0, abs=1e-10)
