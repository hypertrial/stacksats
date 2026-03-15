"""Step definitions for backtest-related features."""

import datetime as dt

import numpy as np
import polars as pl
from pytest_bdd import given, parsers, then, when

from stacksats.backtest import compute_weights_with_features
from stacksats.framework_contract import ALLOCATION_SPAN_DAYS
from stacksats.prelude import backtest_dynamic_dca, compute_cycle_spd, get_backtest_end


def _parse_date(s: str) -> dt.datetime:
    return dt.datetime.strptime(s[:10], "%Y-%m-%d")


# -----------------------------------------------------------------------------
# Given Steps - Backtest Setup
# -----------------------------------------------------------------------------


@given("backtest features are initialized")
def given_backtest_features(sample_features_df, bdd_context):
    """Initialize backtest features context."""
    bdd_context["features_df"] = sample_features_df


@given("a tracking strategy function")
def given_tracking_strategy(bdd_context):
    """Create a strategy function that tracks received dates."""
    received_dates = []

    def tracking_strategy(window_feat):
        received_dates.append(window_feat["date"].max())
        return compute_weights_with_features(
            window_feat,
            features_df=bdd_context["features_df"],
        )

    bdd_context["tracking_strategy"] = tracking_strategy
    bdd_context["received_dates"] = received_dates


@given(parsers.parse('a backtest window from "{start}" to "{end}"'))
def given_backtest_window(start, end, bdd_context):
    """Set up a backtest window."""
    bdd_context["window_start"] = _parse_date(start)
    bdd_context["window_end"] = _parse_date(end)


@given("empty feature window")
def given_empty_window(bdd_context):
    """Create an empty feature window."""
    bdd_context["window_feat"] = pl.DataFrame(schema={"date": pl.Datetime("us")})


@given("single-day feature window")
def given_single_day_window(sample_features_df, bdd_context):
    """Create a contract-valid one-year feature window."""
    single_date = _parse_date("2020-06-01")
    end_date = single_date + dt.timedelta(days=364)
    bdd_context["window_feat"] = sample_features_df.filter(
        (pl.col("date") >= single_date) & (pl.col("date") <= end_date)
    )


# -----------------------------------------------------------------------------
# When Steps - Backtest Actions
# -----------------------------------------------------------------------------


@when("I extract the feature window")
def when_extract_window(bdd_context):
    """Extract feature window for the backtest period."""
    features_df = bdd_context["features_df"]
    start = bdd_context["window_start"]
    end = bdd_context["window_end"]
    bdd_context["window_feat"] = features_df.filter(
        (pl.col("date") >= start) & (pl.col("date") <= end)
    )


@when("I compute weights for the window using explicit feature weights")
def when_compute_weights_with_features(bdd_context):
    """Compute weights using explicit feature inputs."""
    window_feat = bdd_context["window_feat"]
    bdd_context["weights"] = compute_weights_with_features(
        window_feat,
        features_df=bdd_context["features_df"],
    )


@when("I run compute_cycle_spd")
def when_run_cycle_spd(sample_btc_df, bdd_context):
    """Run full SPD computation."""
    features_df = bdd_context["features_df"]
    spd_table = compute_cycle_spd(
        sample_btc_df,
        lambda window_feat: compute_weights_with_features(
            window_feat,
            features_df=features_df,
        ),
        features_df=features_df,
    )
    bdd_context["spd_table"] = spd_table


@when("I run compute_cycle_spd with tracking strategy")
def when_run_cycle_spd_tracking(sample_btc_df, bdd_context):
    """Run SPD computation with tracking strategy."""
    features_df = bdd_context["features_df"]
    tracking_strategy = bdd_context["tracking_strategy"]
    spd_table = compute_cycle_spd(
        sample_btc_df, tracking_strategy, features_df=features_df
    )
    bdd_context["spd_table"] = spd_table


@when("I run backtest_dynamic_dca")
def when_run_backtest_shared(sample_btc_df, bdd_context):
    """Run full backtest."""
    features_df = bdd_context["features_df"]
    spd_table, exp_decay_percentile, uniform_exp_decay_percentile = backtest_dynamic_dca(
        sample_btc_df,
        lambda window_feat: compute_weights_with_features(
            window_feat,
            features_df=features_df,
        ),
        features_df=features_df,
        strategy_label="Test Strategy",
    )
    bdd_context["spd_table"] = spd_table
    bdd_context["exp_decay_percentile"] = exp_decay_percentile
    bdd_context["uniform_exp_decay_percentile"] = uniform_exp_decay_percentile


@when("I compute weights twice for the same window")
def when_compute_weights_twice(bdd_context):
    """Compute weights twice for determinism check."""
    window_feat = bdd_context["window_feat"]
    bdd_context["weights1"] = compute_weights_with_features(
        window_feat,
        features_df=bdd_context["features_df"],
    )
    bdd_context["weights2"] = compute_weights_with_features(
        window_feat,
        features_df=bdd_context["features_df"],
    )


# -----------------------------------------------------------------------------
# Then Steps - Backtest Assertions
# -----------------------------------------------------------------------------


@then("weights should be computed only for the window dates")
def then_weights_match_window(bdd_context):
    """Assert weights match window dates."""
    weights = bdd_context["weights"]
    window_feat = bdd_context["window_feat"]
    assert weights.height == window_feat.height, "Weight count != window size"


@then("weight index should match window dates")
def then_weight_index_matches(bdd_context):
    """Assert weight index matches window dates."""
    weights = bdd_context["weights"]
    start = bdd_context["window_start"]
    end = bdd_context["window_end"]
    assert weights["date"].min() == start, "Weight index start mismatch"
    assert weights["date"].max() == end, "Weight index end mismatch"


@then("no received dates should exceed configured backtest end")
def then_no_future_dates(bdd_context):
    """Assert no dates exceed backtest end."""
    received_dates = bdd_context["received_dates"]
    max_received = max(received_dates)
    backtest_end = _parse_date(get_backtest_end())
    assert max_received <= backtest_end, (
        f"Received date {max_received} > configured backtest end {backtest_end}"
    )


@then("SPD table should not be empty")
def then_spd_not_empty(bdd_context):
    """Assert SPD table has results."""
    spd_table = bdd_context["spd_table"]
    assert spd_table.height > 0, "SPD table is empty"


@then("SPD table should have required columns")
def then_spd_has_columns(bdd_context):
    """Assert SPD table has required columns."""
    spd_table = bdd_context["spd_table"]
    required = [
        "min_sats_per_dollar",
        "max_sats_per_dollar",
        "uniform_sats_per_dollar",
        "dynamic_sats_per_dollar",
        "uniform_percentile",
        "dynamic_percentile",
        "excess_percentile",
    ]
    for col in required:
        assert col in spd_table.columns, f"Missing column: {col}"


@then("percentile values should be in valid range")
def then_percentiles_valid(bdd_context):
    """Assert percentiles are in [0, 100]."""
    spd_table = bdd_context["spd_table"]
    valid_rows = spd_table.drop_nulls(subset=["dynamic_percentile", "uniform_percentile"])
    assert valid_rows["dynamic_percentile"].ge(0).all()
    assert valid_rows["dynamic_percentile"].le(100).all()
    assert valid_rows["uniform_percentile"].ge(0).all()
    assert valid_rows["uniform_percentile"].le(100).all()


@then("excess percentile should equal dynamic minus uniform")
def then_excess_calculation(bdd_context):
    """Assert excess percentile is correctly calculated."""
    spd_table = bdd_context["spd_table"]
    valid_rows = spd_table.drop_nulls(subset=["dynamic_percentile", "uniform_percentile"])
    excess = valid_rows["dynamic_percentile"] - valid_rows["uniform_percentile"]
    np.testing.assert_allclose(
        excess.to_numpy(),
        valid_rows["excess_percentile"].to_numpy(),
        rtol=1e-12,
        atol=1e-12,
    )


@then("both weight computations should be identical")
def then_weights_identical(bdd_context):
    """Assert repeated weight computations are identical."""
    weights1 = bdd_context["weights1"]
    weights2 = bdd_context["weights2"]
    assert weights1.equals(weights2), "Weights are not deterministic"


@then("empty window should produce empty weights")
def then_empty_weights(bdd_context):
    """Assert empty window produces empty weights."""
    weights = bdd_context["weights"]
    assert weights.height == 0, "Expected empty weights"


@then("single-day weight should equal 1.0")
def then_single_day_weight(bdd_context):
    """Assert one-year window weights are valid and normalized."""
    weights = bdd_context["weights"]
    assert weights.height == ALLOCATION_SPAN_DAYS, "Expected configured fixed-span length"
    assert np.isclose(float(weights["weight"].sum()), 1.0), "Weights must sum to 1.0"


@then("exp_decay_percentile should be a number")
def then_exp_decay_is_number(bdd_context):
    """Assert exp_decay_percentile is numeric."""
    exp_decay = bdd_context["exp_decay_percentile"]
    assert isinstance(exp_decay, (int, float)), "exp_decay not numeric"
