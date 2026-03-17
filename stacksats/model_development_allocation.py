"""Allocation-kernel internals for model_development."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from .framework_contract import (
    ALLOCATION_SPAN_DAYS,
    apply_clipped_weight,
    assert_final_invariants,
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
    stable_signal = _compute_stable_signal(raw_arr[:n_past]) if n_past > 0 else np.array([])
    w = np.zeros(n, dtype=float)
    base_weight = 1.0 / n
    locked_prefix = validate_locked_prefix(locked_weights, n_past)
    prefix_len = len(locked_prefix)
    if prefix_len > 0:
        w[:prefix_len] = locked_prefix

    remaining_budget = 1.0 - float(w[:prefix_len].sum())
    for i in range(prefix_len, n_past):
        signal = float(stable_signal[i])
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


def _to_datetime(val: dt.datetime | str) -> dt.datetime:
    if isinstance(val, dt.datetime):
        out = val
    else:
        out = dt.datetime.strptime(str(val)[:10], "%Y-%m-%d")
    if out.tzinfo is not None:
        out = out.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return out.replace(hour=0, minute=0, second=0, microsecond=0)


def _normalize_intent_frame(frame: pl.DataFrame, *, name: str) -> pl.DataFrame:
    """Validate and normalize intent payloads to canonical date/value columns."""
    if not isinstance(frame, pl.DataFrame):
        raise TypeError(f"{name} must be a Polars DataFrame.")
    if "date" not in frame.columns or "value" not in frame.columns:
        raise ValueError(f"{name} must have 'date' and 'value' columns.")
    if frame.is_empty():
        return pl.DataFrame(schema={"date": pl.Datetime("us"), "value": pl.Float64})
    return frame.select(
        pl.col("date").cast(pl.Datetime("us")).dt.replace_time_zone(None).alias("date"),
        pl.col("value").cast(pl.Float64, strict=False).alias("value"),
    )


def _target_profile_to_pl(target_profile: object) -> pl.DataFrame:
    """Normalize TargetProfile or pl.DataFrame to canonical date/value columns."""
    if isinstance(target_profile, pl.DataFrame):
        return _normalize_intent_frame(target_profile, name="target_profile")
    if hasattr(target_profile, "values") and isinstance(
        getattr(target_profile, "values"), pl.DataFrame
    ):
        return _normalize_intent_frame(target_profile.values, name="target_profile")
    raise TypeError("target_profile must be a Polars DataFrame or TargetProfile.")


def _proposals_to_pl(proposals: object) -> pl.DataFrame:
    """Normalize proposals from a Polars DataFrame."""
    if isinstance(proposals, pl.DataFrame):
        return _normalize_intent_frame(proposals, name="proposals")
    raise TypeError("proposals must be a Polars DataFrame.")


def compute_n_past_span(
    start_date: dt.datetime | str,
    end_date: dt.datetime | str,
    current_date: dt.datetime | str,
) -> int:
    """Compute inclusive observed-day count without materializing a date index."""
    start_ts = _to_datetime(start_date)
    end_ts = _to_datetime(end_date)
    current_ts = _to_datetime(current_date)
    if end_ts < start_ts or current_ts < start_ts:
        return 0
    observed_end = min(end_ts, current_ts)
    return (observed_end - start_ts).days + 1


def _calendar_frame(start_ts: dt.datetime, end_ts: dt.datetime) -> pl.DataFrame:
    return pl.DataFrame({
        "date": pl.datetime_range(start_ts, end_ts, interval="1d", eager=True),
    })


def _finite_value_expr(column: str = "_v") -> pl.Expr:
    return (
        pl.col(column)
        .cast(pl.Float64, strict=False)
        .fill_null(0.0)
        .fill_nan(0.0)
    )


def compute_weights_from_target_profile(
    *,
    start_date: dt.datetime | str,
    end_date: dt.datetime | str,
    current_date: dt.datetime | str,
    target_profile: pl.DataFrame | object,
    mode: str = "preference",
    locked_weights: np.ndarray | None = None,
    n_past: int | None = None,
) -> pl.DataFrame:
    """Convert a target profile into final iterative stable allocation weights.

    target_profile must be a Polars DataFrame or TargetProfile with date/value columns.
    Returns pl.DataFrame with columns 'date' and 'weight'.
    """
    target_profile = _target_profile_to_pl(target_profile)
    start_ts = _to_datetime(start_date)
    end_ts = _to_datetime(end_date)
    full_range = _calendar_frame(start_ts, end_ts)
    if full_range.is_empty():
        return pl.DataFrame(schema={"date": pl.Datetime("us"), "weight": pl.Float64})

    n = full_range.height
    base = 1.0 / float(n)

    target_renamed = target_profile.select(["date", pl.col("value").alias("_v")])
    target_df = full_range.join(
        target_renamed,
        on="date",
        how="left",
    ).with_columns(_finite_value_expr("_v").alias("_v"))

    if mode == "absolute":
        target_df = target_df.with_columns(
            pl.col("_v").clip(lower_bound=0.0).alias("_absolute")
        )
        absolute_total = float(target_df["_absolute"].sum() or 0.0)
        if absolute_total <= 0.0:
            target_df = target_df.with_columns(pl.lit(base).alias("_raw"))
        else:
            target_df = target_df.with_columns(
                (pl.col("_absolute") / pl.lit(absolute_total)).alias("_raw")
            )
    elif mode == "preference":
        target_df = target_df.with_columns(
            (pl.lit(base) * pl.col("_v").clip(-50.0, 50.0).exp()).alias("_raw")
        )
    else:
        raise ValueError(f"Unsupported target profile mode '{mode}'.")

    curr_ts = _to_datetime(current_date)
    if n_past is None:
        n_past = compute_n_past_span(start_ts, end_ts, curr_ts)
    raw = target_df["_raw"].to_numpy()
    weights_arr = allocate_sequential_stable(raw, n_past, locked_weights)
    assert_final_invariants(weights_arr)
    return target_df.select("date").with_columns(
        pl.Series("weight", weights_arr.astype(float))
    )


def compute_weights_from_proposals(
    *,
    proposals: pl.DataFrame | object,
    start_date: dt.datetime | str,
    end_date: dt.datetime | str,
    n_past: int,
    locked_weights: np.ndarray | None = None,
) -> pl.DataFrame:
    """Convert per-day user proposals into final framework weights.

    proposals must be a Polars DataFrame with date/value columns.
    Returns pl.DataFrame with columns 'date' and 'weight'.
    """
    proposals = _proposals_to_pl(proposals)
    start_ts = _to_datetime(start_date)
    end_ts = _to_datetime(end_date)
    full_range = _calendar_frame(start_ts, end_ts)
    if full_range.is_empty():
        return pl.DataFrame(schema={"date": pl.Datetime("us"), "weight": pl.Float64})

    prop_renamed = proposals.select(["date", pl.col("value").alias("_v")])
    full_df = full_range.join(
        prop_renamed,
        on="date",
        how="left",
    ).with_columns(_finite_value_expr("_v").alias("_v"))
    proposed_arr = full_df["_v"].to_numpy()

    weights_arr = allocate_from_proposals(
        proposals=proposed_arr,
        n_past=n_past,
        n_total=full_range.height,
        locked_weights=locked_weights,
    )
    assert_final_invariants(weights_arr)
    return full_df.select("date").with_columns(pl.Series("weight", weights_arr.astype(float)))
