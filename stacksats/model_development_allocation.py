"""Allocation-kernel internals for model_development."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .framework_contract import (
    ALLOCATION_SPAN_DAYS,
    apply_clipped_weight,
    assert_final_invariants,
    compute_n_past,
    validate_locked_prefix,
)


def _compute_stable_signal(raw: np.ndarray) -> np.ndarray:
    """Compute stable signal weights using cumulative mean normalization."""
    n = len(raw)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])

    cumsum = np.cumsum(raw)
    running_mean = cumsum / np.arange(1, n + 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        signal = raw / running_mean
    return np.where(np.isfinite(signal), signal, 1.0)


def allocate_sequential_stable(
    raw: np.ndarray,
    n_past: int,
    locked_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Allocate weights with lock-on-compute stability."""
    n = len(raw)
    if n == 0:
        return np.array([])
    enforce_contract_bounds = n == ALLOCATION_SPAN_DAYS
    if n_past <= 0:
        out = np.full(n, 1.0 / n, dtype=float)
        assert_final_invariants(out)
        return out

    n_past = min(n_past, n)
    raw_arr = np.asarray(raw, dtype=float)
    w = np.zeros(n, dtype=float)
    base_weight = 1.0 / n
    locked_prefix = validate_locked_prefix(locked_weights, n_past)
    prefix_len = len(locked_prefix)
    if prefix_len > 0:
        w[:prefix_len] = locked_prefix

    remaining_budget = 1.0 - float(w[:prefix_len].sum())
    for i in range(prefix_len, n_past):
        signal = float(_compute_stable_signal(raw_arr[: i + 1])[-1])
        proposed = signal * base_weight
        clipped, remaining_budget = apply_clipped_weight(
            proposed,
            remaining_budget,
            n - i,
            enforce_contract_bounds=enforce_contract_bounds,
        )
        w[i] = clipped

    n_future = n - n_past
    if n_future > 0:
        uniform_future = max(remaining_budget, 0.0) / n_future
        w[n_past:] = uniform_future
    else:
        w[n - 1] += max(remaining_budget, 0.0)

    assert_final_invariants(w)
    return w


def allocate_from_proposals(
    proposals: np.ndarray,
    n_past: int,
    n_total: int,
    locked_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Allocate final weights from user-proposed per-day values."""
    if n_total == 0:
        return np.array([], dtype=float)
    enforce_contract_bounds = n_total == ALLOCATION_SPAN_DAYS
    if n_past <= 0:
        out = np.full(n_total, 1.0 / n_total, dtype=float)
        assert_final_invariants(out)
        return out

    n_past = min(n_past, n_total)
    proposals_arr = np.asarray(proposals, dtype=float)
    w = np.zeros(n_total, dtype=float)
    locked_prefix = validate_locked_prefix(locked_weights, n_past)
    prefix_len = len(locked_prefix)
    if prefix_len > 0:
        w[:prefix_len] = locked_prefix

    remaining_budget = 1.0 - float(w[:prefix_len].sum())
    for i in range(prefix_len, n_past):
        proposed = float(proposals_arr[i]) if i < len(proposals_arr) else 0.0
        clipped, remaining_budget = apply_clipped_weight(
            proposed,
            remaining_budget,
            n_total - i,
            enforce_contract_bounds=enforce_contract_bounds,
        )
        w[i] = clipped

    n_future = n_total - n_past
    if n_future > 0:
        uniform_future = max(remaining_budget, 0.0) / n_future
        w[n_past:] = uniform_future
    else:
        w[n_total - 1] += max(remaining_budget, 0.0)

    assert_final_invariants(w)
    return w


def compute_weights_from_target_profile(
    *,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    current_date: pd.Timestamp,
    target_profile: pd.Series,
    mode: str = "preference",
    locked_weights: np.ndarray | None = None,
    n_past: int | None = None,
) -> pd.Series:
    """Convert a target profile into final iterative stable allocation weights."""
    full_range = pd.date_range(start=start_date, end=end_date, freq="D")
    if len(full_range) == 0:
        return pd.Series(dtype=float)

    target = target_profile.reindex(full_range)
    target = pd.to_numeric(target, errors="coerce")

    n = len(full_range)
    base = np.ones(n, dtype=float) / n
    if mode == "absolute":
        absolute = target.fillna(0.0).to_numpy(dtype=float)
        absolute = np.where(np.isfinite(absolute), absolute, 0.0)
        absolute = np.clip(absolute, 0.0, None)
        if absolute.sum() <= 0:
            raw = base
        else:
            raw = absolute / absolute.sum()
    elif mode == "preference":
        preference = target.fillna(0.0).to_numpy(dtype=float)
        preference = np.where(np.isfinite(preference), preference, 0.0)
        preference = np.clip(preference, -50, 50)
        raw = base * np.exp(preference)
    else:
        raise ValueError(f"Unsupported target profile mode '{mode}'.")

    if n_past is None:
        n_past = compute_n_past(full_range, current_date)
    weights = allocate_sequential_stable(raw, n_past, locked_weights)
    assert_final_invariants(weights)
    return pd.Series(weights, index=full_range, dtype=float)


def compute_weights_from_proposals(
    *,
    proposals: pd.Series,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    n_past: int,
    locked_weights: np.ndarray | None = None,
) -> pd.Series:
    """Convert per-day user proposals into final framework weights."""
    full_range = pd.date_range(start=start_date, end=end_date, freq="D")
    if len(full_range) == 0:
        return pd.Series(dtype=float)

    proposed = pd.to_numeric(proposals.reindex(full_range), errors="coerce")
    proposed_arr = proposed.fillna(0.0).to_numpy(dtype=float)
    weights = allocate_from_proposals(
        proposals=proposed_arr,
        n_past=n_past,
        n_total=len(full_range),
        locked_weights=locked_weights,
    )
    assert_final_invariants(weights)
    return pd.Series(weights, index=full_range, dtype=float)
