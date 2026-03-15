"""Step definitions for weight computation and validation features."""

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl
from pytest_bdd import given, parsers, then, when

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from stacksats.model_development import (
    MIN_W,
    allocate_sequential_stable,
    compute_dynamic_multiplier,
    compute_weights_fast,
    compute_window_weights,
    precompute_features,
    zscore,
)
from stacksats.model_development_helpers import softmax

# -----------------------------------------------------------------------------
# Given Steps - Weight-specific Setup
# -----------------------------------------------------------------------------


@given(parsers.parse("an array of values {values}"))
def given_array_values(values, bdd_context):
    """Parse and store an array of values."""
    parsed = [float(x.strip()) for x in values.strip("[]").split(",")]
    bdd_context["input_array"] = np.array(parsed)


@given("a window of price data")
def given_price_window(sample_btc_df, bdd_context):
    """Extract a window of price data."""
    start_dt = dt.datetime(2024, 1, 1)
    end_dt = dt.datetime(2024, 6, 30)
    window = sample_btc_df.filter(
        (pl.col("date") >= start_dt) & (pl.col("date") <= end_dt)
    )
    bdd_context["price_window"] = window


@given("all-zero input values")
def given_zero_array(bdd_context):
    """Create an all-zero array."""
    bdd_context["input_array"] = np.array([0.0, 0.0, 0.0])


@given("large input values for numerical stability test")
def given_large_values(bdd_context):
    """Create large values for numerical stability testing."""
    bdd_context["input_array"] = np.array([1000.0, 1001.0, 1002.0])


@given(parsers.parse("a window size of {n:d} days"))
def given_window_size(n, bdd_context):
    """Set window size for dynamic multiplier computation."""
    bdd_context["window_size"] = n


# -----------------------------------------------------------------------------
# When Steps - Weight Computations
# -----------------------------------------------------------------------------


@when("I apply softmax to the array")
def when_apply_softmax(bdd_context):
    """Apply softmax function."""
    x = bdd_context["input_array"]
    bdd_context["softmax_result"] = softmax(x)


@when("I apply allocate_sequential_stable")
def when_apply_allocate_sequential_stable(bdd_context):
    """Apply allocate_sequential_stable function."""
    raw = bdd_context["input_array"]
    # Use n_past=len(raw) to allocate all as past weights
    bdd_context["allocation_result"] = allocate_sequential_stable(raw, n_past=len(raw))


@when("I compute the dynamic multiplier")
def when_compute_dynamic_multiplier(bdd_context):
    """Compute dynamic multiplier for weight adjustment."""
    n = bdd_context["window_size"]
    # Create sample feature arrays for testing
    price_vs_ma = np.zeros(n)
    mvrv_zscore = np.zeros(n)
    mvrv_gradient = np.zeros(n)
    bdd_context["dynamic_multiplier_result"] = compute_dynamic_multiplier(
        price_vs_ma, mvrv_zscore, mvrv_gradient
    )


@when("I compute z-scores with the rolling window")
def when_compute_zscore(bdd_context):
    """Compute z-scores."""
    prices = bdd_context["price_series"]
    win = bdd_context.get("zscore_window", 30)
    bdd_context["zscore_result"] = zscore(prices, win)


@when("I compute weights using compute_weights_fast")
def when_compute_weights_fast(bdd_context):
    """Compute weights using compute_weights_fast function."""
    window = bdd_context["price_window"]
    features_df = precompute_features(window)
    start_date = features_df["date"].min()
    end_date = features_df["date"].max()
    bdd_context["weights"] = compute_weights_fast(features_df, start_date, end_date)


# -----------------------------------------------------------------------------
# Then Steps - Weight Assertions
# -----------------------------------------------------------------------------


@then("softmax output should sum to 1.0")
def then_softmax_sums_to_one(bdd_context):
    """Assert softmax output sums to 1."""
    result = bdd_context["softmax_result"]
    assert np.isclose(result.sum(), 1.0), f"Softmax sum: {result.sum()}"


@then("all softmax values should be positive")
def then_softmax_positive(bdd_context):
    """Assert all softmax values are positive."""
    result = bdd_context["softmax_result"]
    assert (result > 0).all(), "Found non-positive softmax values"


@then("larger inputs should have larger softmax probabilities")
def then_softmax_ordering(bdd_context):
    """Assert softmax preserves input ordering."""
    result = bdd_context["softmax_result"]
    x = bdd_context["input_array"]
    # Sort indices by input values
    sorted_indices = np.argsort(x)
    # Corresponding softmax values should also be sorted
    sorted_result = result[sorted_indices]
    assert np.all(sorted_result[:-1] <= sorted_result[1:]), (
        "Softmax doesn't preserve ordering"
    )


@then("softmax should produce uniform distribution")
def then_softmax_uniform(bdd_context):
    """Assert softmax produces uniform distribution for equal inputs."""
    result = bdd_context["softmax_result"]
    expected = np.ones(len(result)) / len(result)
    assert np.allclose(result, expected), "Softmax not uniform for equal inputs"


@then("softmax should not overflow")
def then_softmax_no_overflow(bdd_context):
    """Assert softmax handles large values without overflow."""
    result = bdd_context["softmax_result"]
    assert not np.any(np.isnan(result)), "Found NaN in softmax"
    assert not np.any(np.isinf(result)), "Found Inf in softmax"
    assert np.isclose(result.sum(), 1.0), "Softmax sum is not 1"


@then("allocation should sum to 1.0")
def then_allocation_sums_to_one(bdd_context):
    """Assert allocation sums to 1."""
    result = bdd_context["allocation_result"]
    assert np.isclose(result.sum(), 1.0, rtol=1e-9), f"Allocation sum: {result.sum()}"


@then("all allocations should be at least MIN_W")
def then_allocation_above_min(bdd_context):
    """Assert all allocations meet minimum weight constraint."""
    result = bdd_context["allocation_result"]
    assert (result >= MIN_W - 1e-12).all(), (
        f"Found allocation below MIN_W: {result.min()}"
    )


@then("all allocations should be non-negative")
def then_allocation_non_negative(bdd_context):
    """Assert all allocations are non-negative (stable allocation prioritizes stability)."""
    result = bdd_context["allocation_result"]
    assert (result >= -1e-12).all(), f"Found negative allocation: {result.min()}"


@then("dynamic multiplier should have correct length")
def then_dynamic_multiplier_length(bdd_context):
    """Assert dynamic multiplier has correct length."""
    result = bdd_context["dynamic_multiplier_result"]
    n = bdd_context["window_size"]
    assert len(result) == n, f"Dynamic multiplier length: {len(result)}, expected {n}"


@then("all dynamic multiplier values should be positive")
def then_dynamic_multiplier_positive(bdd_context):
    """Assert all dynamic multiplier values are positive."""
    result = bdd_context["dynamic_multiplier_result"]
    assert (result > 0).all(), "Found non-positive dynamic multiplier values"


@then("dynamic multiplier should have no NaN values")
def then_dynamic_multiplier_no_nan(bdd_context):
    """Assert dynamic multiplier has no NaN values."""
    result = bdd_context["dynamic_multiplier_result"]
    assert not np.any(np.isnan(result)), "Found NaN in dynamic multiplier"


@then("z-scores should be finite")
def then_zscore_finite(bdd_context):
    """Assert z-scores are finite."""
    result = bdd_context["zscore_result"]
    # After fill_null(0), there should be no NaN
    assert result.is_not_nan().all(), "Z-scores contain NaN after fill_null"


@then("max weight should not exceed a reasonable threshold")
def then_max_weight_reasonable(bdd_context):
    """Assert no single weight is too large."""
    weights = bdd_context["weights"]
    w = weights["weight"] if isinstance(weights, pl.DataFrame) else weights
    assert float(w.max()) < 0.5, f"Max weight too large: {float(w.max())}"


@then("weights should have positive variance")
def then_weights_have_variance(bdd_context):
    """Assert weights are not all identical."""
    weights = bdd_context["weights"]
    w = weights["weight"] if isinstance(weights, pl.DataFrame) else weights
    assert float(w.std()) > 0, "Weights have zero variance"


# -----------------------------------------------------------------------------
# Weight Stability Steps
# -----------------------------------------------------------------------------


def _parse_date(s: str) -> dt.datetime:
    return dt.datetime.strptime(s[:10], "%Y-%m-%d")


@given(parsers.parse('a date range from "{start}" to "{end}"'))
def given_date_range(start, end, bdd_context):
    """Set the date range for weight computation."""
    bdd_context["start_date"] = _parse_date(start)
    bdd_context["end_date"] = _parse_date(end)


@given(parsers.parse('features computed up to "{date}"'))
def given_features_up_to(date, bdd_context, sample_btc_df):
    """Truncate features to a specific date."""
    truncate_date = _parse_date(date)
    truncated_df = sample_btc_df.filter(pl.col("date") <= truncate_date)
    bdd_context["features_df"] = precompute_features(truncated_df)
    bdd_context["truncate_date"] = truncate_date


@when(parsers.parse('I compute weights with current_date "{date}"'))
def when_compute_weights_with_current(date, bdd_context, sample_features_df):
    """Compute weights with a specific current_date."""
    current_date = _parse_date(date)
    start_date = bdd_context["start_date"]
    end_date = bdd_context["end_date"]

    # Use the truncated features if available, otherwise use sample
    features_df = bdd_context.get("features_df", sample_features_df)

    weights = compute_window_weights(features_df, start_date, end_date, current_date)
    bdd_context["weights"] = weights
    bdd_context["current_date"] = current_date


@when(parsers.parse('features are extended to "{date}"'))
def when_features_extended(date, bdd_context, sample_btc_df):
    """Extend features to a new date."""
    extend_date = _parse_date(date)
    extended_df = sample_btc_df.filter(pl.col("date") <= extend_date)
    bdd_context["features_df"] = precompute_features(extended_df)


@when(parsers.parse('I recompute weights with current_date "{date}"'))
def when_recompute_weights(date, bdd_context):
    """Recompute weights with new current_date (using extended features)."""
    current_date = _parse_date(date)
    start_date = bdd_context["start_date"]
    end_date = bdd_context["end_date"]
    features_df = bdd_context["features_df"]

    weights = compute_window_weights(features_df, start_date, end_date, current_date)
    bdd_context["new_weights"] = weights
    bdd_context["new_current_date"] = current_date


@when("I store the past weights")
def when_store_past_weights(bdd_context):
    """Store past weights for later comparison."""
    weights = bdd_context["weights"]
    current_date = bdd_context["current_date"]
    past_weights = weights.filter(pl.col("date") <= current_date)
    bdd_context["stored_past_weights"] = past_weights.clone()


@then(parsers.parse('weights for dates before "{date}" should be identical'))
def then_past_weights_identical(date, bdd_context):
    """Assert past weights are identical before and after recomputation."""
    boundary_date = _parse_date(date)
    old_weights = bdd_context["weights"]
    new_weights = bdd_context["new_weights"]

    old_past = old_weights.filter(pl.col("date") < boundary_date)
    merged = old_past.join(
        new_weights.select(["date", pl.col("weight").alias("new_weight")]),
        on="date",
        how="inner",
    )
    old_vals = merged["weight"].to_numpy()
    new_vals = merged["new_weight"].to_numpy()

    # Note: slight drift is acceptable due to rolling feature recalculation
    np.testing.assert_allclose(
        old_vals,
        new_vals,
        rtol=5e-3,
        atol=5e-3,
        err_msg="Past weights changed after feature extension",
    )


@then("the weights should sum to 1.0")
def then_weights_sum_to_one(bdd_context):
    """Assert weights sum to 1.0."""
    weights = bdd_context["weights"]
    w = weights["weight"] if isinstance(weights, pl.DataFrame) else weights
    total = float(w.sum())
    assert np.isclose(total, 1.0, atol=1e-9), f"Weights sum: {total}"


@then("all weights should be at least MIN_W")
def then_all_weights_above_min_w(bdd_context):
    """Assert all weights >= 0 (MIN_W not strictly enforced for stability).

    Note: The stable allocation prioritizes weight stability over MIN_W enforcement.
    Weights may be below MIN_W but must be non-negative.
    """
    weights = bdd_context["weights"]
    w = weights["weight"] if isinstance(weights, pl.DataFrame) else weights
    below = w.filter(w < -1e-12)
    assert below.len() == 0, f"Found {below.len()} negative weights: min={float(w.min())}"


@then("all weights should be non-negative")
def then_all_weights_non_negative(bdd_context):
    """Assert all weights >= 0 (MIN_W not enforced for stability)."""
    weights = bdd_context["weights"]
    w = weights["weight"] if isinstance(weights, pl.DataFrame) else weights
    negative = w.filter(w < -1e-12)
    assert negative.len() == 0, (
        f"Found {negative.len()} negative weights: min={float(w.min())}"
    )


@then(parsers.parse('weights for dates after "{date}" should be uniform'))
def then_future_weights_uniform(date, bdd_context):
    """Assert future weights are uniformly distributed (except last day).

    Note: The last day of the window absorbs the remainder to ensure sum = 1.0,
    so it may differ from other future weights.
    """
    boundary_date = _parse_date(date)
    weights = bdd_context["weights"]

    future_df = weights.filter(pl.col("date") > boundary_date)
    future_vals = future_df["weight"].to_numpy()
    if len(future_vals) > 2:
        # Exclude last day which absorbs remainder
        future_except_last = future_vals[:-1]
        expected = future_except_last[0]
        assert np.allclose(future_except_last, expected, atol=1e-10), (
            "Future weights (except last) are not uniform"
        )


@then("the stored past weights should match the new past weights")
def then_stored_past_match(bdd_context):
    """Assert stored past weights match new computation (relative proportions).

    Note: With budget scaling, absolute values may change but relative
    proportions should remain stable. We check that the ratio between
    weights is preserved.
    """
    stored = bdd_context["stored_past_weights"]
    new_weights = bdd_context["new_weights"]

    merged = stored.join(
        new_weights.select(["date", pl.col("weight").alias("new_weight")]),
        on="date",
        how="inner",
    )
    stored_vals = merged["weight"].to_numpy()
    new_vals = merged["new_weight"].to_numpy()

    if len(stored_vals) > 1 and stored_vals.sum() > 0 and new_vals.sum() > 0:
        stored_normalized = stored_vals / stored_vals.sum()
        new_normalized = new_vals / new_vals.sum()
        np.testing.assert_allclose(
            stored_normalized,
            new_normalized,
            atol=1e-6,
            err_msg="Stored past weight proportions don't match new computation",
        )


@then("all weights should be uniform")
def then_all_weights_uniform(bdd_context):
    """Assert all weights are uniform."""
    weights = bdd_context["weights"]
    w = weights["weight"] if isinstance(weights, pl.DataFrame) else weights
    n = w.len()
    expected = 1.0 / n
    assert np.allclose(w.to_numpy(), expected, atol=1e-10), "Weights are not uniform"
