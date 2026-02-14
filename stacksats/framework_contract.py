"""Framework-owned allocation contract helpers.

This module centralizes strict invariants for allocation windows and
framework-side allocation mechanics.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

MIN_SPAN_DAYS = 90
MAX_SPAN_DAYS = 1460
DEFAULT_SPAN_DAYS = 365

MIN_DAILY_WEIGHT = 1e-5
MAX_DAILY_WEIGHT = 0.1
SUM_TOLERANCE = 1e-8


def _as_timestamp(value: pd.Timestamp | str) -> pd.Timestamp:
    """Normalize date-like values into timezone-naive timestamps."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def _load_allocation_span_days() -> int:
    """Load and validate fixed allocation span from environment/config defaults."""
    raw = os.getenv("STACKSATS_ALLOCATION_SPAN_DAYS")
    if raw is None:
        span_days = DEFAULT_SPAN_DAYS
    else:
        try:
            span_days = int(raw)
        except ValueError as exc:
            raise ValueError(
                "STACKSATS_ALLOCATION_SPAN_DAYS must be an integer."
            ) from exc
    if span_days < MIN_SPAN_DAYS or span_days > MAX_SPAN_DAYS:
        raise ValueError(
            f"Allocation span must be between {MIN_SPAN_DAYS} and {MAX_SPAN_DAYS} "
            f"days, got {span_days}."
        )
    return span_days


ALLOCATION_SPAN_DAYS = _load_allocation_span_days()
ALLOCATION_WINDOW_OFFSET = pd.Timedelta(days=ALLOCATION_SPAN_DAYS - 1)


def _is_contract_length(n_days: int) -> bool:
    """Return True when n_days equals the configured fixed allocation span."""
    return int(n_days) == ALLOCATION_SPAN_DAYS


def _assert_weight_budget_feasible(n_days: int) -> None:
    """Ensure min/max daily bounds can satisfy a full budget for n_days."""
    if n_days <= 0:
        return
    min_budget = n_days * MIN_DAILY_WEIGHT
    max_budget = n_days * MAX_DAILY_WEIGHT
    if min_budget > 1.0 + SUM_TOLERANCE:
        raise ValueError(
            f"Infeasible allocation bounds: min total {min_budget:.6f} > 1.0 "
            f"for {n_days} days."
        )
    if max_budget < 1.0 - SUM_TOLERANCE:
        raise ValueError(
            f"Infeasible allocation bounds: max total {max_budget:.6f} < 1.0 "
            f"for {n_days} days."
        )


def validate_span_length(
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
) -> int:
    """Validate allocation span cardinality.

    Returns the number of allocation days and raises for invalid spans.
    """
    start_ts = _as_timestamp(start_date)
    end_ts = _as_timestamp(end_date)
    if end_ts < start_ts:
        raise ValueError("end_date must be on or after start_date.")
    n_days = len(pd.date_range(start=start_ts, end=end_ts, freq="D"))
    if n_days != ALLOCATION_SPAN_DAYS:
        raise ValueError(
            "Allocation span must match configured fixed span "
            f"{ALLOCATION_SPAN_DAYS} days, got {n_days}."
        )
    return n_days


def compute_n_past(date_index: pd.DatetimeIndex, current_date: pd.Timestamp | str) -> int:
    """Compute deterministic count of days at-or-before current_date."""
    if len(date_index) == 0:
        return 0
    current_ts = _as_timestamp(current_date)
    normalized_index = pd.DatetimeIndex(date_index).normalize()
    if not normalized_index.is_monotonic_increasing:
        raise ValueError("Allocation index must be monotonic increasing.")
    return int((normalized_index <= current_ts).sum())


def validate_locked_prefix(
    locked_weights: np.ndarray | None,
    n_past: int,
) -> np.ndarray:
    """Validate an immutable locked prefix for the past segment."""
    if locked_weights is None:
        return np.array([], dtype=float)

    locked = np.asarray(locked_weights, dtype=float)
    if locked.ndim != 1:
        raise ValueError("locked_weights must be a 1D array.")
    if len(locked) > n_past:
        raise ValueError(
            f"locked_weights length ({len(locked)}) cannot exceed n_past ({n_past})."
        )
    if not np.isfinite(locked).all():
        raise ValueError("locked_weights must be finite.")
    if (locked < 0).any() or (locked > 1).any():
        raise ValueError("locked_weights values must be within [0, 1].")
    if _is_contract_length(n_past):
        if (locked < MIN_DAILY_WEIGHT - SUM_TOLERANCE).any():
            raise ValueError(
                f"locked_weights contain values below minimum {MIN_DAILY_WEIGHT}."
            )
        if (locked > MAX_DAILY_WEIGHT + SUM_TOLERANCE).any():
            raise ValueError(
                f"locked_weights contain values above maximum {MAX_DAILY_WEIGHT}."
            )

    running_sum = 0.0
    for value in locked:
        running_sum += float(value)
        if running_sum > 1.0 + SUM_TOLERANCE:
            raise ValueError("locked_weights exceed feasible remaining budget.")
    return locked


def apply_clipped_weight(
    proposed_weight: float,
    remaining_budget: float,
    remaining_days_including_today: int = 1,
    *,
    enforce_contract_bounds: bool = False,
) -> tuple[float, float]:
    """Clip a proposed day weight into feasible bounds.

    For contract-valid spans, clipping enforces:
    - per-day [MIN_DAILY_WEIGHT, MAX_DAILY_WEIGHT]
    - feasibility for all future days under remaining budget.
    """
    proposal = float(proposed_weight)
    remaining = max(float(remaining_budget), 0.0)
    if not np.isfinite(proposal):
        proposal = 0.0

    days_left = int(remaining_days_including_today)
    if days_left <= 0:
        return 0.0, remaining

    if enforce_contract_bounds:
        future_days = days_left - 1
        min_future = future_days * MIN_DAILY_WEIGHT
        max_future = future_days * MAX_DAILY_WEIGHT
        lower = max(MIN_DAILY_WEIGHT, remaining - max_future)
        upper = min(MAX_DAILY_WEIGHT, remaining - min_future)
        if lower > upper + SUM_TOLERANCE:
            raise ValueError(
                "No feasible allocation bounds for current day under remaining budget."
            )
        clipped = float(np.clip(proposal, lower, upper))
    else:
        clipped = float(np.clip(proposal, 0.0, remaining))
    return clipped, remaining - clipped


def assert_final_invariants(weights: np.ndarray) -> None:
    """Enforce framework output invariants for final allocations."""
    arr = np.asarray(weights, dtype=float)
    if arr.ndim != 1:
        raise ValueError("weights must be 1D.")
    if len(arr) == 0:
        return
    if not np.isfinite(arr).all():
        raise ValueError("weights contain NaN/inf values.")
    if (arr < -SUM_TOLERANCE).any():
        raise ValueError("weights contain negative values.")
    if (arr > 1.0 + SUM_TOLERANCE).any():
        raise ValueError("weights contain out-of-range values above 1.0.")
    n_days = len(arr)
    if _is_contract_length(n_days):
        _assert_weight_budget_feasible(n_days)
        if (arr < MIN_DAILY_WEIGHT - SUM_TOLERANCE).any():
            raise ValueError(
                f"weights contain values below minimum {MIN_DAILY_WEIGHT}."
            )
        if (arr > MAX_DAILY_WEIGHT + SUM_TOLERANCE).any():
            raise ValueError(
                f"weights contain values above maximum {MAX_DAILY_WEIGHT}."
            )
    total = float(arr.sum())
    if not np.isclose(total, 1.0, atol=SUM_TOLERANCE, rtol=0.0):
        raise ValueError(f"weights must sum to 1.0, got {total:.12f}.")
