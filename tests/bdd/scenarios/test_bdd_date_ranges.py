"""BDD tests for date range generation.

This module wires the date_ranges.feature to step definitions.
"""

import pandas as pd
from pytest_bdd import scenarios

# Import step definitions to make them available
from tests.bdd.step_defs.common_steps import *  # noqa: F401, F403
from tests.bdd.step_defs.database_steps import *  # noqa: F401, F403
from stacksats.framework_contract import ALLOCATION_SPAN_DAYS
from stacksats.prelude import generate_date_ranges

# Load all scenarios from the feature file
scenarios("date_ranges.feature")


def test_bdd_ranges_span_cardinality_matches_config() -> None:
    ranges = generate_date_ranges("2025-01-01", "2027-12-31")
    assert len(ranges) > 0
    for start, end in ranges:
        cardinality = len(pd.date_range(start=start, end=end, freq="D"))
        assert cardinality == ALLOCATION_SPAN_DAYS


def test_bdd_ranges_never_exceed_contract_bounds() -> None:
    ranges = generate_date_ranges("2025-01-01", "2027-12-31")
    assert all(
        len(pd.date_range(start=s, end=e, freq="D")) == ALLOCATION_SPAN_DAYS
        for s, e in ranges
    )
