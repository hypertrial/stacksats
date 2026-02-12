"""Step definitions for data validation and integrity checks."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pytest_bdd import given, then, when

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from stacksats.model_development import MIN_W
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
    key_cols = ["start_date", "end_date", "DCA_date"]
    duplicates = df.duplicated(subset=key_cols, keep=False)
    bdd_context["duplicates"] = duplicates


@when("I check primary key uniqueness")
def when_check_pk(bdd_context):
    """Check primary key uniqueness."""
    df = bdd_context["weights_df"]
    duplicates = df.duplicated(subset=PRIMARY_KEY_COLS, keep=False)
    bdd_context["pk_duplicates"] = duplicates


@when("I check sequential IDs within each range")
def when_check_sequential_ids(bdd_context):
    """Check ID sequentiality."""
    df = bdd_context["weights_df"]
    violations = []
    for (start, end), group in iter_date_ranges(df):
        ids = group["id"].sort_values().values
        expected = np.arange(len(group))
        if not np.array_equal(ids, expected):
            violations.append(f"{start} to {end}")
    bdd_context["id_violations"] = violations


@when("I check date sequentiality within each range")
def when_check_sequential_dates(bdd_context):
    """Check date sequentiality."""
    df = bdd_context["weights_df"]
    violations = []
    for (start, end), group in iter_date_ranges(df):
        dates = pd.to_datetime(group["DCA_date"]).sort_values()
        gaps = dates.diff().dropna()
        invalid = gaps[gaps != pd.Timedelta(days=1)]
        if not invalid.empty:
            violations.append(f"{start} to {end}")
    bdd_context["date_violations"] = violations


@when("I check row counts per range")
def when_check_row_counts(bdd_context):
    """Check row counts match expected."""
    df = bdd_context["weights_df"]
    violations = []
    for (start, end), group in iter_date_ranges(df):
        expected = get_range_days(start, end)
        if len(group) != expected:
            violations.append(
                f"{start} to {end}: got {len(group)}, expected {expected}"
            )
    bdd_context["count_violations"] = violations


@when("I check for missing dates in ranges")
def when_check_missing_dates(bdd_context):
    """Check for missing dates."""
    df = bdd_context["weights_df"]
    violations = []
    for (start, end), group in iter_date_ranges(df):
        expected = set(pd.date_range(start=start, end=end, freq="D"))
        actual = set(pd.to_datetime(group["DCA_date"]))
        missing = expected - actual
        if missing:
            violations.append(f"{start} to {end}: missing {len(missing)} dates")
    bdd_context["missing_dates_violations"] = violations


@when("I check date ordering constraints")
def when_check_date_ordering(bdd_context):
    """Check start < end for all rows."""
    df = bdd_context["weights_df"]
    starts = pd.to_datetime(df["start_date"])
    ends = pd.to_datetime(df["end_date"])
    invalid = df[starts >= ends]
    bdd_context["ordering_violations"] = invalid


@when("I check DCA dates are within range")
def when_check_dca_in_range(bdd_context):
    """Check DCA dates are within [start, end]."""
    df = bdd_context["weights_df"].copy()
    for col in DATE_COLS:
        df[f"_{col}"] = pd.to_datetime(df[col])

    invalid = df[
        (df["_DCA_date"] < df["_start_date"]) | (df["_DCA_date"] > df["_end_date"])
    ]
    bdd_context["dca_range_violations"] = invalid


@when("I check weight sum per range")
def when_check_weight_sums(bdd_context):
    """Check weights sum to 1.0 per range."""
    df = bdd_context["weights_df"]
    violations = []
    for (start, end), group in iter_date_ranges(df):
        weight_sum = group["weight"].sum()
        if not np.isclose(weight_sum, 1.0, atol=WEIGHT_SUM_TOLERANCE):
            violations.append(f"{start} to {end}: sum={weight_sum:.10f}")
    bdd_context["weight_sum_violations"] = violations


@when("I check data types")
def when_check_dtypes(bdd_context):
    """Check data types match schema."""
    df = bdd_context["weights_df"]
    dtype_issues = []

    if df["id"].dtype not in [np.int64, np.int32, int]:
        dtype_issues.append(f"id: {df['id'].dtype}")

    if df["weight"].dtype not in [np.float64, np.float32, float]:
        dtype_issues.append(f"weight: {df['weight'].dtype}")

    non_null_btc = df["btc_usd"].dropna()
    if len(non_null_btc) > 0:
        if non_null_btc.dtype not in [np.float64, np.float32, float]:
            dtype_issues.append(f"btc_usd: {non_null_btc.dtype}")

    bdd_context["dtype_issues"] = dtype_issues


@when("I check for null values in required columns")
def when_check_nulls(bdd_context):
    """Check for null values in required columns."""
    df = bdd_context["weights_df"]
    null_issues = []

    if df["id"].isna().any():
        null_issues.append(f"id: {df['id'].isna().sum()} nulls")

    if df["weight"].isna().any():
        null_issues.append(f"weight: {df['weight'].isna().sum()} nulls")

    for col in DATE_COLS:
        if df[col].isna().any():
            null_issues.append(f"{col}: {df[col].isna().sum()} nulls")

    bdd_context["null_issues"] = null_issues


# -----------------------------------------------------------------------------
# Then Steps - Validation Assertions
# -----------------------------------------------------------------------------


@then("there should be no duplicate rows")
def then_no_duplicates(bdd_context):
    """Assert no duplicate rows."""
    duplicates = bdd_context["duplicates"]
    assert not duplicates.any(), f"Found {duplicates.sum() // 2} duplicate rows"


@then("primary keys should be unique")
def then_pk_unique(bdd_context):
    """Assert primary keys are unique."""
    pk_duplicates = bdd_context["pk_duplicates"]
    assert not pk_duplicates.any(), f"Found {pk_duplicates.sum()} duplicate PKs"


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
    assert invalid.empty, f"Found {len(invalid)} rows with start >= end"


@then("DCA_date should be within the range")
def then_dca_in_range(bdd_context):
    """Assert DCA dates are within range."""
    invalid = bdd_context["dca_range_violations"]
    assert invalid.empty, f"Found {len(invalid)} DCA_dates outside range"


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
    below_min = df[df["weight"] < MIN_W - FLOAT_TOLERANCE]
    assert below_min.empty, f"Found {len(below_min)} weights below MIN_W"


@then("all weights should be non-negative")
def then_weights_non_negative(bdd_context):
    """Assert weights are non-negative."""
    df = bdd_context["weights_df"]
    negative = df[df["weight"] < 0]
    assert negative.empty, f"Found {len(negative)} negative weights"


@then("all weights should be finite")
def then_weights_finite_validation(bdd_context):
    """Assert weights are finite."""
    df = bdd_context["weights_df"]
    assert df["weight"].notna().all(), "Found NaN weights"
    assert np.isfinite(df["weight"]).all(), "Found non-finite weights"


@then("weights should have variance")
def then_weights_have_variance(bdd_context):
    """Assert weights are not all identical."""
    df = bdd_context["weights_df"]
    for (start, end), group in iter_date_ranges(df):
        if len(group) > 1:
            assert group["weight"].std() > 0, f"Range {start} to {end}: zero variance"
