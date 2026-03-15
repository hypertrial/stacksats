"""Shared helper functions used by StrategyRunner."""

from __future__ import annotations

import datetime as dt
import hashlib
import json

import numpy as np
import polars as pl

from .framework_contract import ALLOCATION_SPAN_DAYS
from .strategy_types import TargetProfile

DATE_COL = "date"


def _value_col(df: pl.DataFrame) -> str:
    """Return the value/weight column name for comparison."""
    for c in ("weight", "value", "preference"):
        if c in df.columns:
            return c
    return df.columns[-1] if df.columns else ""


def weights_match(lhs: pl.DataFrame, rhs: pl.DataFrame, *, atol: float = 1e-12) -> bool:
    """Compare two DataFrames (date, weight/value) for equality."""
    if lhs.is_empty() and rhs.is_empty():
        return True
    if lhs.is_empty() or rhs.is_empty():
        return False
    lcol = _value_col(lhs)
    rcol = _value_col(rhs)
    if not lcol or not rcol:
        return False
    merged = lhs.select([DATE_COL, pl.col(lcol).alias("l")]).join(
        rhs.select([DATE_COL, pl.col(rcol).alias("r")]),
        on=DATE_COL,
        how="outer",
    )
    l_vals = merged["l"].fill_null(0.0).to_numpy()
    r_vals = merged["r"].fill_null(0.0).to_numpy()
    return bool(
        np.all(np.isfinite(l_vals))
        and np.all(np.isfinite(r_vals))
        and np.allclose(l_vals, r_vals, rtol=0.0, atol=atol)
    )


def profile_values(profile: TargetProfile | pl.DataFrame) -> pl.DataFrame:
    """Extract values DataFrame from profile or return as-is if already DataFrame."""
    if isinstance(profile, TargetProfile):
        return profile.values
    return profile


def frame_signature(df: pl.DataFrame) -> tuple:
    """Return a hashable signature for a DataFrame."""
    try:
        payload = json.dumps(
            {
                "cols": list(df.columns),
                "dtypes": [str(df[c].dtype) for c in df.columns],
                "rows": df.to_dicts(),
            },
            sort_keys=True,
            default=str,
        )
        row_hash = int(hashlib.sha256(payload.encode()).hexdigest()[:16], 16)
    except (TypeError, ValueError):
        row_hash = hash(str(df.to_dicts()))
    return (
        row_hash,
        tuple(str(c) for c in df.columns),
        tuple(str(df[c].dtype) for c in df.columns),
        (df.height, len(df.columns)),
    )


def perturb_future_features(
    features_df: pl.DataFrame, probe: dt.datetime
) -> pl.DataFrame:
    """Perturb future rows (date > probe) in features for leakage testing."""
    perturbed = features_df.clone()
    future_mask = pl.col(DATE_COL) > probe
    future = perturbed.filter(future_mask)
    if future.is_empty():
        return perturbed

    numeric_cols = [
        c
        for c in future.columns
        if c != DATE_COL and future[c].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
    ]
    if numeric_cols:
        n = perturbed.height
        mask_arr = (perturbed[DATE_COL] > probe).to_numpy()
        n_future = int(mask_arr.sum())
        ramp = np.linspace(1.0, 2.0, n_future, dtype=float)
        for col in numeric_cols:
            arr = perturbed[col].to_numpy().astype(float)
            future_vals = arr[mask_arr]
            perturbed_vals = np.where(
                np.isfinite(future_vals), (-3.0 * future_vals) + ramp, 0.0
            )
            arr[mask_arr] = perturbed_vals
            perturbed = perturbed.with_columns(pl.Series(col, arr))

    non_numeric = [c for c in future.columns if c not in numeric_cols and c != DATE_COL]
    if non_numeric and future.height > 1:
        for col in non_numeric:
            rev_vals = future[col].reverse().to_list()
            arr = perturbed[col].to_list()
            mask_arr = (perturbed[DATE_COL] > probe).to_numpy()
            idx = 0
            for i in range(n):
                if mask_arr[i]:
                    arr[i] = rev_vals[idx]
                    idx += 1
            perturbed = perturbed.with_columns(pl.Series(col, arr))
    return perturbed


def perturb_future_source_data(
    btc_df: pl.DataFrame, probe: dt.datetime
) -> pl.DataFrame:
    """Perturb future source rows while preserving the observed prefix."""
    perturbed = btc_df.clone()
    future_mask = pl.col(DATE_COL) > probe
    future = perturbed.filter(future_mask)
    if future.is_empty():
        return perturbed

    numeric_cols = [
        c
        for c in future.columns
        if c != DATE_COL and future[c].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
    ]
    if numeric_cols:
        mask_arr = (perturbed[DATE_COL] > probe).to_numpy()
        n_future = int(mask_arr.sum())
        ramp = np.linspace(0.5, 1.5, n_future, dtype=float)
        for col in numeric_cols:
            arr = perturbed[col].to_numpy().astype(float)
            future_vals = arr[mask_arr]
            perturbed_vals = np.where(
                np.isfinite(future_vals), (-2.0 * future_vals) + ramp, 0.0
            )
            arr[mask_arr] = perturbed_vals
            perturbed = perturbed.with_columns(pl.Series(col, arr))
    return perturbed


def build_fold_ranges(
    start_ts: dt.datetime,
    end_ts: dt.datetime,
) -> list[tuple[dt.datetime, dt.datetime]]:
    """Build fold boundaries for walk-forward validation."""
    all_days = pl.datetime_range(start_ts, end_ts, interval="1d", eager=True)
    if all_days.len() < (ALLOCATION_SPAN_DAYS * 2):
        return []
    max_folds = min(4, all_days.len() // ALLOCATION_SPAN_DAYS)
    boundaries = np.linspace(0, all_days.len(), num=max_folds + 1, dtype=int)
    folds: list[tuple[dt.datetime, dt.datetime]] = []
    for i in range(max_folds):
        left = int(boundaries[i])
        right = int(boundaries[i + 1]) - 1
        if right <= left:
            continue
        fold_start = all_days[left]
        fold_end = all_days[right]
        span = (fold_end - fold_start).days + 1 if hasattr(fold_end - fold_start, "days") else right - left + 1
        if span >= ALLOCATION_SPAN_DAYS:
            folds.append((fold_start, fold_end))
    return folds
