"""Common step definitions shared across all feature files.

This module contains Given/When/Then steps that are reused across multiple
feature files for setup, actions, and assertions.
"""

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from pytest_bdd import given, parsers, then, when

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from stacksats.model_development import (
    MIN_W,
    compute_weights_fast,
    precompute_features,
)


def _parse_date(s: str) -> dt.datetime:
    """Parse date string to naive datetime."""
    return dt.datetime.strptime(s[:10], "%Y-%m-%d")

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
    bdd_context["start_date"] = _parse_date(start_date)
    bdd_context["end_date"] = _parse_date(end_date)


@given(parsers.parse('current date is "{current_date}"'))
def given_current_date(current_date, bdd_context):
    """Set the current date for testing."""
    bdd_context["current_date"] = _parse_date(current_date)


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


def _weights_series(weights) -> pl.Series:
    """Extract weight series from DataFrame or Series."""
    if isinstance(weights, pl.DataFrame):
        return weights["weight"]
    return weights


@then("the weights should sum to 1.0")
def then_weights_sum_to_one(bdd_context):
    """Assert weights sum to 1.0."""
    weights = bdd_context["weights"]
    w = _weights_series(weights)
    total = float(w.sum())
    assert np.isclose(total, 1.0, rtol=1e-9, atol=1e-12), (
        f"Weights sum to {total:.15f}, expected 1.0"
    )


@then("all weights should be at least MIN_W")
def then_weights_above_min(bdd_context):
    """Assert all weights are at least MIN_W."""
    weights = bdd_context["weights"]
    w = _weights_series(weights)
    below_min = w.filter(w < MIN_W - 1e-12)
    assert below_min.len() == 0, (
        f"Found {below_min.len()} weights below MIN_W ({MIN_W}). "
        f"Min weight: {float(w.min()):.2e}"
    )


@then("all weights should be positive")
def then_weights_positive(bdd_context):
    """Assert all weights are non-negative."""
    weights = bdd_context["weights"]
    w = _weights_series(weights)
    assert w.ge(0).all(), "Found negative weights"


@then("all weights should be finite")
def then_weights_finite(bdd_context):
    """Assert all weights are finite (no NaN or Inf)."""
    weights = bdd_context["weights"]
    w = _weights_series(weights)
    assert w.is_not_nan().all(), "Found NaN weights"
    assert np.isfinite(w.to_numpy()).all(), "Found non-finite weights"


@then("the weights should be deterministic")
def then_weights_deterministic(bdd_context):
    """Assert weights are deterministic (same input = same output)."""
    features_df = bdd_context["features_df"]
    start_date = bdd_context["start_date"]
    end_date = bdd_context["end_date"]

    weights1 = bdd_context["weights"]
    weights2 = compute_weights_fast(features_df, start_date, end_date)

    w1 = _weights_series(weights1)
    w2 = _weights_series(weights2)
    assert w1.len() == w2.len()
    np.testing.assert_allclose(w1.to_numpy(), w2.to_numpy(), rtol=1e-12, atol=1e-12)


@then(parsers.parse("the weight count should be {expected_count:d}"))
def then_weight_count(expected_count, bdd_context):
    """Assert the number of weights matches expected."""
    weights = bdd_context["weights"]
    n = weights.height if isinstance(weights, pl.DataFrame) else weights.len()
    assert n == expected_count, (
        f"Weight count {n} != expected {expected_count}"
    )


@then("no weights should be NaN")
def then_no_nan_weights(bdd_context):
    """Assert no NaN values in weights."""
    weights = bdd_context["weights"]
    w = _weights_series(weights)
    assert w.is_not_nan().all(), f"Found {w.is_null().sum()} NaN weights"


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
        assert not features_df[feat].is_null().any(), f"NaN values in {feat}"
