"""Shared helper functions used by StrategyRunner."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from ..data.windowing import (
    DATE_COL,
    build_window_bounds,
    build_window_index,
    resolve_window_bounds,
    slice_window_or_filter,
)
from ..features.materialization import hash_dataframe
from ..framework_contract import ALLOCATION_SPAN_DAYS
from ..strategy_types import TargetProfile


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
    merged = lhs.select([DATE_COL, pl.col(lcol).cast(pl.Float64, strict=False).alias("l")]).join(
        rhs.select([DATE_COL, pl.col(rcol).cast(pl.Float64, strict=False).alias("r")]),
        on=DATE_COL,
        how="full",
        coalesce=True,
    )
    stats = merged.select(
        (
            (~pl.col("l").fill_null(0.0).is_finite())
            | (~pl.col("r").fill_null(0.0).is_finite())
        ).any().alias("has_non_finite"),
        (
            (pl.col("l").fill_null(0.0) - pl.col("r").fill_null(0.0))
            .abs()
            .max()
        ).alias("max_diff"),
    ).row(0, named=True)
    return (
        not bool(stats["has_non_finite"])
        and float(stats["max_diff"] or 0.0) <= float(atol)
    )


def profile_values(profile: TargetProfile | pl.DataFrame) -> pl.DataFrame:
    """Extract values DataFrame from profile or return as-is if already DataFrame."""
    if isinstance(profile, TargetProfile):
        return profile.values
    return profile


def frame_signature(df: pl.DataFrame) -> tuple:
    """Return a hashable signature for a DataFrame."""
    row_hash = int(hash_dataframe(df)[:16], 16)
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
        n_future = future.height
        denom = max(n_future - 1, 1)
        perturbed = perturbed.with_row_index("__row").with_columns(
            pl.when(pl.col(DATE_COL) > probe)
            .then(
                1.0
                + ((pl.col("__row") - (perturbed.height - n_future)) / float(denom))
            )
            .otherwise(None)
            .alias("__future_ramp")
        )
        perturbed = perturbed.with_columns(
            [
                pl.when(pl.col(DATE_COL) > probe)
                .then(
                    pl.when(pl.col(col).cast(pl.Float64, strict=False).is_finite())
                    .then((-3.0 * pl.col(col).cast(pl.Float64, strict=False)) + pl.col("__future_ramp"))
                    .otherwise(0.0)
                )
                .otherwise(pl.col(col))
                .alias(col)
                for col in numeric_cols
            ]
        ).drop(["__row", "__future_ramp"])

    non_numeric = [c for c in future.columns if c not in numeric_cols and c != DATE_COL]
    if non_numeric and future.height > 1:
        n = perturbed.height
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
        n_future = future.height
        denom = max(n_future - 1, 1)
        perturbed = perturbed.with_row_index("__row").with_columns(
            pl.when(pl.col(DATE_COL) > probe)
            .then(
                0.5
                + ((pl.col("__row") - (perturbed.height - n_future)) / float(denom))
            )
            .otherwise(None)
            .alias("__future_ramp")
        )
        perturbed = perturbed.with_columns(
            [
                pl.when(pl.col(DATE_COL) > probe)
                .then(
                    pl.when(pl.col(col).cast(pl.Float64, strict=False).is_finite())
                    .then((-2.0 * pl.col(col).cast(pl.Float64, strict=False)) + pl.col("__future_ramp"))
                    .otherwise(0.0)
                )
                .otherwise(pl.col(col))
                .alias(col)
                for col in numeric_cols
            ]
        ).drop(["__row", "__future_ramp"])
    return perturbed


def build_fold_ranges(
    start_ts: dt.datetime,
    end_ts: dt.datetime,
) -> list[tuple[dt.datetime, dt.datetime]]:
    """Build fold boundaries for walk-forward validation."""
    all_days = pl.datetime_range(start_ts, end_ts, interval="1d", eager=True)
    total_days = all_days.len()
    if total_days < (ALLOCATION_SPAN_DAYS * 2):
        return []
    max_folds = min(4, total_days // ALLOCATION_SPAN_DAYS)
    boundaries = np.asarray(np.linspace(0, total_days, num=max_folds + 1, dtype=int), dtype=int)
    if boundaries.shape[0] != (max_folds + 1):
        return []
    if (
        boundaries[0] != 0
        or boundaries[-1] != total_days
        or np.any(np.diff(boundaries) <= 0)
    ):
        boundaries = np.array(
            [(idx * total_days) // max_folds for idx in range(max_folds + 1)],
            dtype=int,
        )
    folds: list[tuple[dt.datetime, dt.datetime]] = []
    for i in range(max_folds):
        left = int(boundaries[i])
        right = int(boundaries[i + 1]) - 1
        if right > left:
            fold_start = all_days[left]
            fold_end = all_days[right]
            span = (fold_end - fold_start).days + 1 if hasattr(fold_end - fold_start, "days") else right - left + 1
            if span >= ALLOCATION_SPAN_DAYS:
                folds.append((fold_start, fold_end))
    return folds


__all__ = [
    "DATE_COL",
    "build_window_bounds",
    "build_window_index",
    "resolve_window_bounds",
    "slice_window_or_filter",
    "weights_match",
    "profile_values",
    "frame_signature",
    "perturb_future_features",
    "perturb_future_source_data",
    "build_fold_ranges",
]
