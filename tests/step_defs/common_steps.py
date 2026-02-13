"""Common step definitions shared across all feature files.

This module contains Given/When/Then steps that are reused across multiple
feature files for setup, actions, and assertions.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pytest_bdd import given, parsers, then, when

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from stacksats.model_development import (
    MIN_W,
    compute_weights_fast,
    precompute_features,
)

# -----------------------------------------------------------------------------
# Fixtures for BDD tests (re-exported from conftest.py)
# -----------------------------------------------------------------------------


@pytest.fixture
def bdd_context():
    """Shared context dictionary for passing data between steps."""
    return {}


# -----------------------------------------------------------------------------
# Given Steps - Setup and Data Preparation
# -----------------------------------------------------------------------------


@given("sample BTC price data from 2020 to 2025")
def given_sample_btc_data(sample_btc_df, bdd_context):
    """Provide sample BTC price data."""
    bdd_context["btc_df"] = sample_btc_df
    return sample_btc_df


@given("precomputed features from the price data")
def given_precomputed_features(sample_features_df, bdd_context):
    """Provide precomputed features."""
    bdd_context["features_df"] = sample_features_df
    return sample_features_df


@given(parsers.parse('a date range from "{start_date}" to "{end_date}"'))
def given_date_range(start_date, end_date, bdd_context):
    """Set up a date range for testing."""
    bdd_context["start_date"] = pd.Timestamp(start_date)
    bdd_context["end_date"] = pd.Timestamp(end_date)


@given(parsers.parse('current date is "{current_date}"'))
def given_current_date(current_date, bdd_context):
    """Set the current date for testing."""
    bdd_context["current_date"] = pd.Timestamp(current_date)


@given("a mock database connection")
def given_mock_db(mock_db_connection, bdd_context):
    """Provide a mock database connection."""
    mock_conn, mock_cursor = mock_db_connection
    bdd_context["mock_conn"] = mock_conn
    bdd_context["mock_cursor"] = mock_cursor


@given("sample weights data")
def given_sample_weights(sample_weights_df, bdd_context):
    """Provide sample weights DataFrame."""
    bdd_context["weights_df"] = sample_weights_df


@given("sample SPD backtest results")
def given_sample_spd(sample_spd_df, bdd_context):
    """Provide sample SPD DataFrame."""
    bdd_context["spd_df"] = sample_spd_df


# -----------------------------------------------------------------------------
# When Steps - Actions
# -----------------------------------------------------------------------------


@when("I compute weights for the date range")
def when_compute_weights(bdd_context):
    """Compute weights for the configured date range."""
    features_df = bdd_context["features_df"]
    start_date = bdd_context["start_date"]
    end_date = bdd_context["end_date"]

    weights = compute_weights_fast(features_df, start_date, end_date)
    bdd_context["weights"] = weights


@when("I precompute features")
def when_precompute_features(bdd_context):
    """Precompute features from the BTC DataFrame."""
    btc_df = bdd_context["btc_df"]
    features = precompute_features(btc_df)
    bdd_context["features_df"] = features


# -----------------------------------------------------------------------------
# Then Steps - Assertions
# -----------------------------------------------------------------------------


@then("the weights should sum to 1.0")
def then_weights_sum_to_one(bdd_context):
    """Assert weights sum to 1.0."""
    weights = bdd_context["weights"]
    assert np.isclose(weights.sum(), 1.0, rtol=1e-9, atol=1e-12), (
        f"Weights sum to {weights.sum():.15f}, expected 1.0"
    )


@then("all weights should be at least MIN_W")
def then_weights_above_min(bdd_context):
    """Assert all weights are at least MIN_W."""
    weights = bdd_context["weights"]
    below_min = weights[weights < MIN_W - 1e-12]
    assert below_min.empty, (
        f"Found {len(below_min)} weights below MIN_W ({MIN_W}). "
        f"Min weight: {weights.min():.2e}"
    )


@then("all weights should be positive")
def then_weights_positive(bdd_context):
    """Assert all weights are non-negative."""
    weights = bdd_context["weights"]
    assert (weights >= 0).all(), "Found negative weights"


@then("all weights should be finite")
def then_weights_finite(bdd_context):
    """Assert all weights are finite (no NaN or Inf)."""
    weights = bdd_context["weights"]
    assert weights.notna().all(), "Found NaN weights"
    assert np.isfinite(weights).all(), "Found non-finite weights"


@then("the weights should be deterministic")
def then_weights_deterministic(bdd_context):
    """Assert weights are deterministic (same input = same output)."""
    features_df = bdd_context["features_df"]
    start_date = bdd_context["start_date"]
    end_date = bdd_context["end_date"]

    weights1 = bdd_context["weights"]
    weights2 = compute_weights_fast(features_df, start_date, end_date)

    pd.testing.assert_series_equal(weights1, weights2, check_names=False)


@then(parsers.parse("the weight count should be {expected_count:d}"))
def then_weight_count(expected_count, bdd_context):
    """Assert the number of weights matches expected."""
    weights = bdd_context["weights"]
    assert len(weights) == expected_count, (
        f"Weight count {len(weights)} != expected {expected_count}"
    )


@then("no weights should be NaN")
def then_no_nan_weights(bdd_context):
    """Assert no NaN values in weights."""
    weights = bdd_context["weights"]
    assert weights.notna().all(), f"Found {weights.isna().sum()} NaN weights"


@then("the features should contain all required columns")
def then_features_have_columns(bdd_context):
    """Assert features DataFrame has all required columns."""
    from stacksats.model_development import FEATS

    features_df = bdd_context["features_df"]
    for feat in FEATS:
        assert feat in features_df.columns, f"Missing feature column: {feat}"


@then("the features should have no NaN values")
def then_features_no_nan(bdd_context):
    """Assert features have no NaN values."""
    from stacksats.model_development import FEATS

    features_df = bdd_context["features_df"]
    for feat in FEATS:
        assert not features_df[feat].isna().any(), f"NaN values in {feat}"
