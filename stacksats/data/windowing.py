"""Shared date-window slicing helpers used by data and runner flows."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import polars as pl

DATE_COL = "date"


@dataclass(frozen=True)
class WindowBounds:
    """Resolved row bounds for a requested date window."""

    start_idx: int | None
    end_idx: int | None
    can_slice: bool


@dataclass(frozen=True)
class WindowPlan:
    """Polars-native date lookup plan for window slicing."""

    frame: pl.DataFrame
    lookup: pl.DataFrame
    has_unique_dates: bool


def _window_bound_col(prefix: str, name: str) -> str:
    return f"{prefix}_{name}" if prefix else name


def _calendar_window_length_expr(start_col: str, end_col: str) -> pl.Expr:
    return (pl.col(end_col) - pl.col(start_col)).dt.total_days() + 1


def build_window_index(frame: pl.DataFrame) -> tuple[pl.DataFrame, WindowPlan]:
    """Return a date-sorted frame and Polars-native lookup plan for window slicing."""
    sorted_frame = frame.sort(DATE_COL)
    indexed = sorted_frame.with_row_index("__row_nr")
    lookup = indexed.group_by(DATE_COL, maintain_order=True).agg(
        pl.col("__row_nr").first().alias("__row_nr"),
        pl.len().alias("__count"),
    ).sort(DATE_COL)
    has_unique_dates = bool((lookup["__count"] == 1).all()) if not lookup.is_empty() else True
    return sorted_frame, WindowPlan(
        frame=sorted_frame,
        lookup=lookup,
        has_unique_dates=has_unique_dates,
    )


def build_window_bounds(
    plan: WindowPlan,
    windows: pl.DataFrame,
    *,
    start_col: str = "window_start",
    end_col: str = "window_end",
    expected_days: int | None = None,
    prefix: str = "",
) -> pl.DataFrame:
    """Join requested windows against a plan's row lookup table."""
    start_idx_col = _window_bound_col(prefix, "start_idx")
    end_idx_col = _window_bound_col(prefix, "end_idx")
    can_slice_col = _window_bound_col(prefix, "can_slice")
    start_count_col = _window_bound_col(prefix, "start_count")
    end_count_col = _window_bound_col(prefix, "end_count")
    expected_len = (
        pl.lit(int(expected_days))
        if expected_days is not None
        else _calendar_window_length_expr(start_col, end_col)
    )
    start_lookup = plan.lookup.rename({
        DATE_COL: start_col,
        "__row_nr": start_idx_col,
        "__count": start_count_col,
    })
    end_lookup = plan.lookup.rename({
        DATE_COL: end_col,
        "__row_nr": end_idx_col,
        "__count": end_count_col,
    })
    return windows.join(start_lookup, on=start_col, how="left").join(
        end_lookup,
        on=end_col,
        how="left",
    ).with_columns(
        (
            pl.lit(plan.has_unique_dates)
            & pl.col(start_idx_col).is_not_null()
            & pl.col(end_idx_col).is_not_null()
            & (pl.col(start_count_col) == 1)
            & (pl.col(end_count_col) == 1)
            & (pl.col(end_idx_col) >= pl.col(start_idx_col))
            & ((pl.col(end_idx_col) - pl.col(start_idx_col) + 1) == expected_len)
        ).alias(can_slice_col)
    )


def resolve_window_bounds(
    plan: WindowPlan,
    start: dt.datetime,
    end: dt.datetime,
    *,
    expected_days: int | None = None,
) -> WindowBounds:
    """Resolve slice bounds for a single window request."""
    bounds = build_window_bounds(
        plan,
        pl.DataFrame({"window_start": [start], "window_end": [end]}),
        expected_days=expected_days,
    ).row(0, named=True)
    start_idx = bounds.get("start_idx")
    end_idx = bounds.get("end_idx")
    return WindowBounds(
        start_idx=int(start_idx) if start_idx is not None else None,
        end_idx=int(end_idx) if end_idx is not None else None,
        can_slice=bool(bounds["can_slice"]),
    )


def slice_window_or_filter(
    frame: pl.DataFrame,
    date_index: WindowPlan | dict[dt.datetime, int],
    start: dt.datetime,
    end: dt.datetime,
    *,
    expected_days: int | None = None,
) -> pl.DataFrame:
    """Slice contiguous daily windows by position, falling back to date filtering."""
    if isinstance(date_index, WindowPlan):
        bounds = resolve_window_bounds(
            date_index,
            start,
            end,
            expected_days=expected_days,
        )
        if bounds.can_slice and bounds.start_idx is not None and bounds.end_idx is not None:
            return frame.slice(bounds.start_idx, bounds.end_idx - bounds.start_idx + 1)
    else:
        start_idx = date_index.get(start)
        if start_idx is not None:
            if expected_days is not None:
                window = frame.slice(start_idx, expected_days)
                if window.height == expected_days and window[DATE_COL][-1] == end:
                    return window
            else:
                end_idx = date_index.get(end)
                if end_idx is not None and end_idx >= start_idx:
                    return frame.slice(start_idx, end_idx - start_idx + 1)
    return frame.filter((pl.col(DATE_COL) >= start) & (pl.col(DATE_COL) <= end))


__all__ = [
    "DATE_COL",
    "WindowBounds",
    "WindowPlan",
    "build_window_bounds",
    "build_window_index",
    "resolve_window_bounds",
    "slice_window_or_filter",
]
