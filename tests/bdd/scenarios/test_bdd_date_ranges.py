"""BDD tests for date range generation."""

from stacksats.framework_contract import ALLOCATION_SPAN_DAYS
from stacksats.prelude import generate_date_ranges


def test_bdd_ranges_span_cardinality_matches_config() -> None:
    ranges = generate_date_ranges("2025-01-01", "2027-12-31")
    assert len(ranges) > 0
    for start, end in ranges:
        cardinality = (end - start).days + 1
        assert cardinality == ALLOCATION_SPAN_DAYS


def test_bdd_ranges_never_exceed_contract_bounds() -> None:
    ranges = generate_date_ranges("2025-01-01", "2027-12-31")
    assert all(
        ((e - s).days + 1) == ALLOCATION_SPAN_DAYS
        for s, e in ranges
    )
