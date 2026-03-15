"""Step definitions for data validation and integrity checks."""

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl
from pytest_bdd import given, then, when

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from stacksats.model_development import MIN_W
from stacksats.prelude import date_range_list
from tests.test_helpers import (
    DATE_COLS,
    FLOAT_TOLERANCE,
    PRIMARY_KEY_COLS,
    WEIGHT_SUM_TOLERANCE,
    get_range_days,
    iter_date_ranges,
)

# -----------------------------------------------------------------------------
# Given Steps - Validation Setup
# -----------------------------------------------------------------------------


@given("the sample weights DataFrame")
def given_weights_df(sample_weights_df, bdd_context):
    """Provide sample weights DataFrame for validation."""
    bdd_context["weights_df"] = sample_weights_df


# -----------------------------------------------------------------------------
# When Steps - Validation Actions
# -----------------------------------------------------------------------------


@when("I check for duplicate rows")
def when_check_duplicates(bdd_context):
    """Check for duplicate rows."""
    df = bdd_context["weights_df"]
    key_cols = ["start_date", "end_date", "date"]
    dup_struct = pl.struct([pl.col(c) for c in key_cols])
    duplicates = df.filter(dup_struct.is_duplicated())
    bdd_context["duplicates"] = duplicates


@when("I check primary key uniqueness")
def when_check_pk(bdd_context):
    """Check primary key uniqueness."""
    df = bdd_context["weights_df"]
    dup_struct = pl.struct([pl.col(c) for c in PRIMARY_KEY_COLS])
    duplicates = df.filter(dup_struct.is_duplicated())
    bdd_context["pk_duplicates"] = duplicates


@when("I check sequential IDs within each range")
def when_check_sequential_ids(bdd_context):
    """Check day_index sequentiality."""
    df = bdd_context["weights_df"]
    violations = []
    for (start, end), group in iter_date_ranges(df):
        ids = group["day_index"].sort().to_numpy()
        expected = np.arange(group.height)
        if not np.array_equal(ids, expected):
            violations.append(f"{start} to {end}")
    bdd_context["id_violations"] = violations


@when("I check date sequentiality within each range")
def when_check_sequential_dates(bdd_context):
    """Check date sequentiality."""
    df = bdd_context["weights_df"]
    violations = []
    one_day = dt.timedelta(days=1)
    for (start, end), group in iter_date_ranges(df):
        dates = group["date"].sort()
        if dates.dtype == pl.Utf8:
            dates = dates.str.to_datetime()
        gaps = dates.diff().drop_nulls()
        invalid = gaps.filter(gaps != one_day)
        if invalid.len() > 0:
            violations.append(f"{start} to {end}")
    bdd_context["date_violations"] = violations


@when("I check row counts per range")
def when_check_row_counts(bdd_context):
    """Check row counts match expected."""
    df = bdd_context["weights_df"]
    violations = []
    for (start, end), group in iter_date_ranges(df):
        expected = get_range_days(str(start)[:10], str(end)[:10])
        if group.height != expected:
            violations.append(
                f"{start} to {end}: got {group.height}, expected {expected}"
            )
    bdd_context["count_violations"] = violations


@when("I check for missing dates in ranges")
def when_check_missing_dates(bdd_context):
    """Check for missing dates."""
    df = bdd_context["weights_df"]
    violations = []
    for (start, end), group in iter_date_ranges(df):
        start_s = str(start)[:10]
        end_s = str(end)[:10]
        expected = {str(d)[:10] for d in date_range_list(start_s, end_s)}
        actual = {str(d)[:10] for d in group["date"].to_list()}
        missing = expected - actual
        if missing:
            violations.append(f"{start} to {end}: missing {len(missing)} dates")
    bdd_context["missing_dates_violations"] = violations


@when("I check date ordering constraints")
def when_check_date_ordering(bdd_context):
    """Check start < end for all rows."""
    df = bdd_context["weights_df"]
    invalid = df.filter(pl.col("start_date") >= pl.col("end_date"))
    bdd_context["ordering_violations"] = invalid


@when("I check DCA dates are within range")
def when_check_dca_in_range(bdd_context):
    """Check DCA dates are within [start, end]."""
    df = bdd_context["weights_df"]
    invalid = df.filter(
        (pl.col("date") < pl.col("start_date")) | (pl.col("date") > pl.col("end_date"))
    )
    bdd_context["dca_range_violations"] = invalid


@when("I check weight sum per range")
def when_check_weight_sums(bdd_context):
    """Check weights sum to 1.0 per range."""
    df = bdd_context["weights_df"]
    violations = []
    for (start, end), group in iter_date_ranges(df):
        weight_sum = float(group["weight"].sum())
        if not np.isclose(weight_sum, 1.0, atol=WEIGHT_SUM_TOLERANCE):
            violations.append(f"{start} to {end}: sum={weight_sum:.10f}")
    bdd_context["weight_sum_violations"] = violations


@when("I check data types")
def when_check_dtypes(bdd_context):
    """Check data types match schema."""
    df = bdd_context["weights_df"]
    dtype_issues = []

    if df["day_index"].dtype not in [pl.Int64, pl.Int32]:
        dtype_issues.append(f"day_index: {df['day_index'].dtype}")

    if df["weight"].dtype not in [pl.Float64, pl.Float32]:
        dtype_issues.append(f"weight: {df['weight'].dtype}")

    non_null_btc = df["price_usd"].drop_nulls()
    if non_null_btc.len() > 0:
        if non_null_btc.dtype not in [pl.Float64, pl.Float32]:
            dtype_issues.append(f"price_usd: {non_null_btc.dtype}")

    bdd_context["dtype_issues"] = dtype_issues


@when("I check for null values in required columns")
def when_check_nulls(bdd_context):
    """Check for null values in required columns."""
    df = bdd_context["weights_df"]
    null_issues = []

    if df["day_index"].is_null().any():
        null_issues.append(f"day_index: {df['day_index'].null_count()} nulls")

    if df["weight"].is_null().any():
        null_issues.append(f"weight: {df['weight'].null_count()} nulls")

    for col in DATE_COLS:
        if df[col].is_null().any():
            null_issues.append(f"{col}: {df[col].null_count()} nulls")

    bdd_context["null_issues"] = null_issues


# -----------------------------------------------------------------------------
# Then Steps - Validation Assertions
# -----------------------------------------------------------------------------


@then("there should be no duplicate rows")
def then_no_duplicates(bdd_context):
    """Assert no duplicate rows."""
    duplicates = bdd_context["duplicates"]
    assert duplicates.height == 0, f"Found {duplicates.height} duplicate rows"


@then("primary keys should be unique")
def then_pk_unique(bdd_context):
    """Assert primary keys are unique."""
    pk_duplicates = bdd_context["pk_duplicates"]
    assert pk_duplicates.height == 0, f"Found {pk_duplicates.height} duplicate PKs"


@then("IDs should be sequential within each range")
def then_ids_sequential(bdd_context):
    """Assert IDs are sequential."""
    violations = bdd_context["id_violations"]
    assert not violations, f"Non-sequential IDs in: {violations}"


@then("dates should be sequential within each range")
def then_dates_sequential(bdd_context):
    """Assert dates are sequential."""
    violations = bdd_context["date_violations"]
    assert not violations, f"Non-sequential dates in: {violations}"


@then("row counts should match expected")
def then_row_counts_match(bdd_context):
    """Assert row counts match."""
    violations = bdd_context["count_violations"]
    assert not violations, f"Row count violations: {violations}"


@then("there should be no missing dates")
def then_no_missing_dates(bdd_context):
    """Assert no missing dates."""
    violations = bdd_context["missing_dates_violations"]
    assert not violations, f"Missing dates: {violations}"


@then("start_date should be before end_date")
def then_start_before_end(bdd_context):
    """Assert start < end."""
    invalid = bdd_context["ordering_violations"]
    assert invalid.height == 0, f"Found {invalid.height} rows with start >= end"


@then("date should be within the range")
def then_dca_in_range(bdd_context):
    """Assert DCA dates are within range."""
    invalid = bdd_context["dca_range_violations"]
    assert invalid.height == 0, f"Found {invalid.height} dates outside range"


@then("weights should sum to 1.0 per range")
def then_weights_sum_per_range(bdd_context):
    """Assert weights sum to 1.0."""
    violations = bdd_context["weight_sum_violations"]
    assert not violations, f"Weight sum violations: {violations}"


@then("data types should match schema")
def then_dtypes_match(bdd_context):
    """Assert data types match."""
    issues = bdd_context["dtype_issues"]
    assert not issues, f"Data type issues: {issues}"


@then("required columns should have no null values")
def then_no_nulls(bdd_context):
    """Assert no nulls in required columns."""
    issues = bdd_context["null_issues"]
    assert not issues, f"Null value issues: {issues}"


@then("all weights should be above minimum")
def then_weights_above_min_validation(bdd_context):
    """Assert weights above MIN_W."""
    df = bdd_context["weights_df"]
    below_min = df.filter(pl.col("weight") < MIN_W - FLOAT_TOLERANCE)
    assert below_min.height == 0, f"Found {below_min.height} weights below MIN_W"


@then("all weights should be non-negative")
def then_weights_non_negative(bdd_context):
    """Assert weights are non-negative."""
    df = bdd_context["weights_df"]
    negative = df.filter(pl.col("weight") < 0)
    assert negative.height == 0, f"Found {negative.height} negative weights"


@then("all weights should be finite")
def then_weights_finite_validation(bdd_context):
    """Assert weights are finite."""
    df = bdd_context["weights_df"]
    assert df["weight"].is_not_nan().all(), "Found NaN weights"
    assert np.isfinite(df["weight"].to_numpy()).all(), "Found non-finite weights"


@then("weights should have variance")
def then_weights_have_variance(bdd_context):
    """Assert weights are not all identical."""
    df = bdd_context["weights_df"]
    for (start, end), group in iter_date_ranges(df):
        if group.height > 1:
            assert float(group["weight"].std()) > 0, f"Range {start} to {end}: zero variance"
