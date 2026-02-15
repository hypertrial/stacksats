from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import stacksats.framework_contract as framework_contract
from stacksats.framework_contract import (
    ALLOCATION_SPAN_DAYS,
    apply_clipped_weight,
    assert_final_invariants,
    compute_n_past,
    validate_span_length,
    validate_locked_prefix,
)


def test_compute_n_past_handles_timezone_aware_current_date() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    current_date = pd.Timestamp("2024-01-03 22:00:00", tz="UTC")

    result = compute_n_past(idx, current_date)

    assert result == 3


def test_compute_n_past_returns_zero_for_empty_index() -> None:
    idx = pd.DatetimeIndex([])
    assert compute_n_past(idx, "2024-01-01") == 0


def test_compute_n_past_rejects_non_monotonic_index() -> None:
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-01")])

    with pytest.raises(ValueError, match="monotonic increasing"):
        compute_n_past(idx, "2024-01-02")


def test_validate_locked_prefix_none_returns_empty_array() -> None:
    result = validate_locked_prefix(None, n_past=3)

    assert isinstance(result, np.ndarray)
    assert result.size == 0


def test_validate_locked_prefix_rejects_non_1d() -> None:
    with pytest.raises(ValueError, match="1D array"):
        validate_locked_prefix(np.array([[0.1, 0.2]]), n_past=2)


def test_validate_locked_prefix_rejects_length_above_n_past() -> None:
    with pytest.raises(ValueError, match="cannot exceed n_past"):
        validate_locked_prefix(np.array([0.1, 0.2, 0.3]), n_past=2)


def test_validate_locked_prefix_rejects_non_finite() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        validate_locked_prefix(np.array([0.1, np.inf]), n_past=2)


def test_validate_locked_prefix_rejects_values_outside_zero_one() -> None:
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        validate_locked_prefix(np.array([-0.1, 0.2]), n_past=2)


def test_validate_locked_prefix_rejects_infeasible_running_sum() -> None:
    with pytest.raises(ValueError, match="exceed feasible remaining budget"):
        validate_locked_prefix(np.array([0.8, 0.3]), n_past=2)


def test_apply_clipped_weight_handles_non_finite_proposal() -> None:
    clipped, remaining = apply_clipped_weight(float("inf"), 0.7)

    assert clipped == 0.0
    assert remaining == 0.7


def test_apply_clipped_weight_clamps_negative_remaining_budget() -> None:
    clipped, remaining = apply_clipped_weight(0.5, -1.0)

    assert clipped == 0.0
    assert remaining == 0.0


def test_apply_clipped_weight_raises_when_enforced_bounds_are_infeasible() -> None:
    with pytest.raises(ValueError, match="No feasible allocation bounds"):
        apply_clipped_weight(
            proposed_weight=0.0,
            remaining_budget=0.0,
            remaining_days_including_today=2,
            enforce_contract_bounds=True,
        )


def test_apply_clipped_weight_returns_zero_when_no_days_left() -> None:
    clipped, remaining = apply_clipped_weight(
        proposed_weight=0.3,
        remaining_budget=0.7,
        remaining_days_including_today=0,
        enforce_contract_bounds=True,
    )

    assert clipped == 0.0
    assert remaining == 0.7


def test_load_allocation_span_days_rejects_non_integer_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STACKSATS_ALLOCATION_SPAN_DAYS", "not-an-int")

    with pytest.raises(ValueError, match="must be an integer"):
        framework_contract._load_allocation_span_days()


def test_load_allocation_span_days_rejects_out_of_range_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "STACKSATS_ALLOCATION_SPAN_DAYS",
        str(framework_contract.MIN_SPAN_DAYS - 1),
    )

    with pytest.raises(ValueError, match="Allocation span must be between"):
        framework_contract._load_allocation_span_days()


def test_assert_weight_budget_feasible_returns_for_non_positive_days() -> None:
    assert framework_contract._assert_weight_budget_feasible(0) is None


def test_assert_weight_budget_feasible_rejects_min_budget_overflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(framework_contract, "MIN_DAILY_WEIGHT", 2.0)
    with pytest.raises(ValueError, match="min total"):
        framework_contract._assert_weight_budget_feasible(1)


def test_assert_weight_budget_feasible_rejects_max_budget_underflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(framework_contract, "MAX_DAILY_WEIGHT", 0.2)
    with pytest.raises(ValueError, match="max total"):
        framework_contract._assert_weight_budget_feasible(1)


def test_validate_span_length_rejects_end_before_start() -> None:
    with pytest.raises(ValueError, match="on or after"):
        validate_span_length("2024-01-02", "2024-01-01")


def test_validate_locked_prefix_rejects_values_below_contract_minimum() -> None:
    locked = np.full(ALLOCATION_SPAN_DAYS, 1.0 / ALLOCATION_SPAN_DAYS, dtype=float)
    locked[0] = framework_contract.MIN_DAILY_WEIGHT / 10.0

    with pytest.raises(ValueError, match="below minimum"):
        validate_locked_prefix(locked, n_past=ALLOCATION_SPAN_DAYS)


def test_validate_locked_prefix_rejects_values_above_contract_maximum() -> None:
    locked = np.full(ALLOCATION_SPAN_DAYS, 1.0 / ALLOCATION_SPAN_DAYS, dtype=float)
    locked[0] = framework_contract.MAX_DAILY_WEIGHT + 1e-6

    with pytest.raises(ValueError, match="above maximum"):
        validate_locked_prefix(locked, n_past=ALLOCATION_SPAN_DAYS)


def test_assert_final_invariants_rejects_non_1d() -> None:
    with pytest.raises(ValueError, match="weights must be 1D"):
        assert_final_invariants(np.array([[0.5, 0.5]]))


def test_assert_final_invariants_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="NaN/inf"):
        assert_final_invariants(np.array([0.5, np.nan]))


def test_assert_final_invariants_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="negative values"):
        assert_final_invariants(np.array([0.8, -0.1, 0.3]))


def test_assert_final_invariants_rejects_values_above_one() -> None:
    with pytest.raises(ValueError, match="above 1.0"):
        assert_final_invariants(np.array([1.1, -1e-12]))


def test_assert_final_invariants_rejects_sum_mismatch() -> None:
    with pytest.raises(ValueError, match="must sum to 1.0"):
        assert_final_invariants(np.array([0.2, 0.2]))


def test_assert_final_invariants_accepts_empty_array() -> None:
    assert_final_invariants(np.array([], dtype=float))


def test_assert_final_invariants_rejects_below_min_for_contract_span() -> None:
    base = 1.0 / ALLOCATION_SPAN_DAYS
    arr = np.full(ALLOCATION_SPAN_DAYS, base, dtype=float)
    arr[0] = framework_contract.MIN_DAILY_WEIGHT / 10.0
    deficit = base - arr[0]
    arr[1:] += deficit / (ALLOCATION_SPAN_DAYS - 1)

    with pytest.raises(ValueError, match="below minimum"):
        assert_final_invariants(arr)


def test_assert_final_invariants_rejects_above_max_for_contract_span() -> None:
    base = 1.0 / ALLOCATION_SPAN_DAYS
    arr = np.full(ALLOCATION_SPAN_DAYS, base, dtype=float)
    arr[0] = framework_contract.MAX_DAILY_WEIGHT + 1e-6
    excess = arr[0] - base
    arr[1:] -= excess / (ALLOCATION_SPAN_DAYS - 1)

    with pytest.raises(ValueError, match="above maximum"):
        assert_final_invariants(arr)
