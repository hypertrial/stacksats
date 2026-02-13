"""Framework-owned allocation contract helpers.

This module centralizes strict invariants for allocation windows and
framework-side allocation mechanics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ALLOWED_SPAN_DAYS = (365, 366)
SUM_TOLERANCE = 1e-8


def _as_timestamp(value: pd.Timestamp | str) -> pd.Timestamp:
    """Normalize date-like values into timezone-naive timestamps."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


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
    if n_days not in ALLOWED_SPAN_DAYS:
        raise ValueError(
            f"Allocation span must have 365 or 366 allocation days, got {n_days}."
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

    running_sum = 0.0
    for value in locked:
        running_sum += float(value)
        if running_sum > 1.0 + SUM_TOLERANCE:
            raise ValueError("locked_weights exceed feasible remaining budget.")
    return locked


def apply_clipped_weight(proposed_weight: float, remaining_budget: float) -> tuple[float, float]:
    """Clip a proposed day weight into feasible bounds."""
    proposal = float(proposed_weight)
    remaining = max(float(remaining_budget), 0.0)
    if not np.isfinite(proposal):
        proposal = 0.0
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
    total = float(arr.sum())
    if not np.isclose(total, 1.0, atol=SUM_TOLERANCE, rtol=0.0):
        raise ValueError(f"weights must sum to 1.0, got {total:.12f}.")
