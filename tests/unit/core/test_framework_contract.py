from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stacksats.framework_contract import (
    apply_clipped_weight,
    assert_final_invariants,
    compute_n_past,
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
