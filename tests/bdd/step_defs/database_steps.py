"""Step definitions for database and export operations."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pytest_bdd import given, parsers, then, when

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from stacksats.export_weights import (
    DATE_FREQ,
    MIN_RANGE_LENGTH_DAYS,
    RANGE_END,
    RANGE_START,
    create_table_if_not_exists,
    generate_date_ranges,
    get_db_connection,
    group_ranges_by_start_date,
    process_start_date_batch,
    table_is_empty,
    today_data_exists,
)
from tests.test_helpers import PRICE_COL

# -----------------------------------------------------------------------------
# Given Steps - Database Setup
# -----------------------------------------------------------------------------


@given("an empty mock database table")
def given_empty_table(mock_db_connection, bdd_context):
    """Configure mock DB as empty."""
    mock_conn, mock_cursor = mock_db_connection
    mock_cursor.fetchone.return_value = (0,)
    bdd_context["mock_conn"] = mock_conn
    bdd_context["mock_cursor"] = mock_cursor


@given("a non-empty mock database table")
def given_non_empty_table(mock_db_connection, bdd_context):
    """Configure mock DB with data."""
    mock_conn, mock_cursor = mock_db_connection
    mock_cursor.fetchone.return_value = (100,)
    bdd_context["mock_conn"] = mock_conn
    bdd_context["mock_cursor"] = mock_cursor


@given("DATABASE_URL is not set")
def given_no_db_url(mocker, bdd_context):
    """Clear DATABASE_URL environment variable."""
    mocker.patch.dict(os.environ, {}, clear=True)
    if "DATABASE_URL" in os.environ:
        del os.environ["DATABASE_URL"]


@given(parsers.parse("today's data count is {count:d}"))
def given_today_data_count(count, bdd_context):
    """Set today's data count in mock."""
    mock_cursor = bdd_context["mock_cursor"]
    mock_cursor.fetchone.return_value = (count,)


@given("default date range configuration")
def given_default_date_config(bdd_context):
    """Store default date range config."""
    bdd_context["range_start"] = RANGE_START
    bdd_context["range_end"] = RANGE_END
    bdd_context["min_range_length"] = MIN_RANGE_LENGTH_DAYS


@given(
    parsers.parse('date range from "{start}" to "{end}" with min length {min_len:d}')
)
def given_custom_date_range(start, end, min_len, bdd_context):
    """Set custom date range configuration."""
    bdd_context["range_start"] = start
    bdd_context["range_end"] = end
    bdd_context["min_range_length"] = min_len


# -----------------------------------------------------------------------------
# When Steps - Database Actions
# -----------------------------------------------------------------------------


@when("I generate date ranges")
def when_generate_ranges(bdd_context):
    """Generate date ranges."""
    ranges = generate_date_ranges(
        bdd_context["range_start"],
        bdd_context["range_end"],
        bdd_context["min_range_length"],
    )
    bdd_context["date_ranges"] = ranges


@when("I group ranges by start date")
def when_group_ranges(bdd_context):
    """Group date ranges by start date."""
    ranges = bdd_context["date_ranges"]
    grouped = group_ranges_by_start_date(ranges)
    bdd_context["grouped_ranges"] = grouped


@when("I create the table if not exists")
def when_create_table(bdd_context):
    """Call create_table_if_not_exists."""
    mock_conn = bdd_context["mock_conn"]
    create_table_if_not_exists(mock_conn)


@when("I check if the table is empty")
def when_check_table_empty(bdd_context):
    """Check if table is empty."""
    mock_conn = bdd_context["mock_conn"]
    result = table_is_empty(mock_conn)
    bdd_context["table_is_empty"] = result


@when(parsers.parse('I check if data exists for "{date}"'))
def when_check_today_data(date, bdd_context):
    """Check if today's data exists."""
    mock_conn = bdd_context["mock_conn"]
    result = today_data_exists(mock_conn, date)
    bdd_context["data_exists"] = result


@when("I get a database connection without DATABASE_URL")
def when_get_connection_no_url(bdd_context):
    """Attempt to get DB connection without URL."""
    try:
        get_db_connection()
        bdd_context["connection_error"] = None
    except ValueError as e:
        bdd_context["connection_error"] = e


@when("I get a database connection with valid URL")
def when_get_connection_valid(bdd_context):
    """Get DB connection with valid URL."""
    mock_conn = bdd_context["mock_conn"]
    bdd_context["connection"] = mock_conn


@when("I process a start date batch")
def when_process_batch(sample_features_df, sample_btc_df, bdd_context):
    """Process a start date batch."""
    start_date = bdd_context.get("start_date", pd.Timestamp("2024-01-07"))
    # Handle both end_date (singular) and end_dates (plural)
    if "end_dates" in bdd_context:
        end_dates = bdd_context["end_dates"]
    elif "end_date" in bdd_context:
        end_dates = [bdd_context["end_date"]]
    else:
        end_dates = [pd.Timestamp("2025-01-05")]
    current_date = bdd_context.get("current_date", pd.Timestamp("2025-01-05"))

    result = process_start_date_batch(
        start_date,
        end_dates,
        sample_features_df,
        sample_btc_df,
        current_date,
        PRICE_COL,
    )
    bdd_context["batch_result"] = result


# -----------------------------------------------------------------------------
# Then Steps - Database Assertions
# -----------------------------------------------------------------------------


@then("date ranges should not be empty")
def then_ranges_not_empty(bdd_context):
    """Assert date ranges are not empty."""
    ranges = bdd_context["date_ranges"]
    assert len(ranges) > 0, "Date ranges are empty"


@then("all ranges should have 1-year span")
def then_ranges_one_year(bdd_context):
    """Assert all ranges are exactly 1 year."""
    ranges = bdd_context["date_ranges"]
    for start, end in ranges:
        cardinality = len(pd.date_range(start=start, end=end, freq="D"))
        assert cardinality in (
            365,
            366,
        ), f"Range has {cardinality} allocation days, expected 365-366"


@then("all ranges should be within configured bounds")
def then_ranges_in_bounds(bdd_context):
    """Assert all ranges are within bounds."""
    ranges = bdd_context["date_ranges"]
    range_start = pd.Timestamp(bdd_context["range_start"])
    range_end = pd.Timestamp(bdd_context["range_end"])
    for start, end in ranges:
        assert start >= range_start, f"Start {start} < range_start"
        assert end <= range_end, f"End {end} > range_end"


@then("start dates should be sequential daily")
def then_sequential_starts(bdd_context):
    """Assert start dates are sequential."""
    ranges = bdd_context["date_ranges"]
    for i in range(len(ranges) - 1):
        start1, _ = ranges[i]
        start2, _ = ranges[i + 1]
        assert (start2 - start1).days == 1, "Start dates not sequential"


@then("grouped ranges should preserve all end dates")
def then_grouped_preserves_ends(bdd_context):
    """Assert grouping preserves all end dates."""
    ranges = bdd_context["date_ranges"]
    grouped = bdd_context["grouped_ranges"]
    total_ranges = sum(len(ends) for ends in grouped.values())
    assert total_ranges == len(ranges), "Grouping lost some ranges"


@then("date ranges should be empty")
def then_ranges_empty(bdd_context):
    """Assert date ranges are empty."""
    ranges = bdd_context["date_ranges"]
    assert len(ranges) == 0, f"Expected empty ranges, got {len(ranges)}"


@then("table_is_empty should return True")
def then_table_is_empty_true(bdd_context):
    """Assert table is empty."""
    result = bdd_context["table_is_empty"]
    assert result is True, "table_is_empty should be True"


@then("table_is_empty should return False")
def then_table_is_empty_false(bdd_context):
    """Assert table is not empty."""
    result = bdd_context["table_is_empty"]
    assert result is False, "table_is_empty should be False"


@then("data_exists should return True")
def then_data_exists_true(bdd_context):
    """Assert data exists."""
    result = bdd_context["data_exists"]
    assert result is True, "data_exists should be True"


@then("data_exists should return False")
def then_data_exists_false(bdd_context):
    """Assert data doesn't exist."""
    result = bdd_context["data_exists"]
    assert result is False, "data_exists should be False"


@then("a ValueError should be raised for missing DATABASE_URL")
def then_db_url_error(bdd_context):
    """Assert ValueError was raised."""
    error = bdd_context["connection_error"]
    assert error is not None, "Expected ValueError"
    assert "DATABASE_URL" in str(error), "Error should mention DATABASE_URL"


@then("CREATE TABLE should be executed")
def then_create_executed(bdd_context):
    """Assert CREATE TABLE was called."""
    mock_cursor = bdd_context["mock_cursor"]
    assert mock_cursor.execute.called, "execute not called"


@then("commit should be called")
def then_commit_called(bdd_context):
    """Assert commit was called."""
    mock_conn = bdd_context["mock_conn"]
    assert mock_conn.commit.called, "commit not called"


@then("batch result should have required columns")
def then_batch_has_columns(bdd_context):
    """Assert batch result has required columns."""
    result = bdd_context["batch_result"]
    required = ["id", "start_date", "end_date", "DCA_date", "btc_usd", "weight"]
    for col in required:
        assert col in result.columns, f"Missing column: {col}"


@then("batch weights should sum to 1.0")
def then_batch_weights_sum(bdd_context):
    """Assert batch weights sum to 1."""
    result = bdd_context["batch_result"]
    weight_sum = result["weight"].sum()
    assert np.isclose(weight_sum, 1.0, rtol=1e-6), (
        f"Weights sum to {weight_sum}, expected 1.0"
    )


@then("DATE_FREQ should be daily")
def then_date_freq_daily(bdd_context):
    """Assert DATE_FREQ is daily."""
    assert DATE_FREQ == "D", f"DATE_FREQ is {DATE_FREQ}, expected 'D'"
