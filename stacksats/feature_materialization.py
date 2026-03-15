"""Helpers for causal feature materialization and stable snapshot hashing."""

from __future__ import annotations

import datetime as dt
import hashlib
import json

import polars as pl


VERSIONED_FEATURE_REQUIRED_COLUMNS = (
    "effective_date",
    "available_at",
)

DATE_COL = "date"


def _normalized_daily_expr(frame: pl.DataFrame, column: str) -> pl.Expr:
    """Return a canonical daily datetime expression for a date-like column."""
    dtype = frame[column].dtype
    if dtype == pl.Utf8:
        base = pl.col(column).str.to_datetime(strict=False)
    elif dtype == pl.Date:
        base = pl.col(column).cast(pl.Datetime)
    else:
        base = pl.col(column).cast(pl.Datetime, strict=False)
    return base.dt.replace_time_zone(None).dt.truncate("1d")


def normalize_timestamp(value: dt.datetime | str) -> dt.datetime:
    """Normalize date-like values into timezone-naive daily timestamps."""
    if isinstance(value, str):
        value = dt.datetime.strptime(value[:10], "%Y-%m-%d")
    out = value.replace(hour=0, minute=0, second=0, microsecond=0)
    if out.tzinfo is not None:
        out = out.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return out


def build_observed_frame(
    features_df: pl.DataFrame,
    *,
    start_date: dt.datetime | str,
    current_date: dt.datetime | str,
) -> pl.DataFrame:
    """Return a sorted copy restricted to the observed prefix."""
    if features_df.is_empty():
        start_ts = normalize_timestamp(start_date)
        current_ts = normalize_timestamp(current_date)
        dates = pl.datetime_range(start_ts, current_ts, interval="1d", eager=True)
        return pl.DataFrame({DATE_COL: dates})

    if DATE_COL not in features_df.columns:
        raise ValueError(f"features_df must have '{DATE_COL}' column.")

    start_ts = normalize_timestamp(start_date)
    current_ts = normalize_timestamp(current_date)
    if current_ts < start_ts:
        return pl.DataFrame(schema=features_df.schema)

    frame = features_df.clone()
    frame = frame.with_columns(_normalized_daily_expr(frame, DATE_COL).alias(DATE_COL))
    frame = frame.unique(subset=[DATE_COL], keep="last").sort(DATE_COL)
    return frame.filter(
        (pl.col(DATE_COL) >= start_ts) & (pl.col(DATE_COL) <= current_ts)
    )


def materialize_versioned_observations(
    observations: pl.DataFrame,
    *,
    as_of_date: dt.datetime | str,
    effective_date_col: str = "effective_date",
    available_at_col: str = "available_at",
    revision_col: str = "revision_id",
) -> pl.DataFrame:
    """Select the latest available revision for each effective date."""
    if observations.is_empty():
        return pl.DataFrame()

    missing = [
        column
        for column in VERSIONED_FEATURE_REQUIRED_COLUMNS
        if column not in observations.columns
    ]
    if missing:
        raise ValueError(
            "Versioned observation table is missing required columns: "
            f"{missing}."
        )

    as_of_ts = normalize_timestamp(as_of_date)
    frame = observations.clone()
    for col in (effective_date_col, available_at_col):
        frame = frame.with_columns(_normalized_daily_expr(frame, col).alias(col))
    frame = frame.filter(pl.col(available_at_col) <= as_of_ts)
    if frame.is_empty():
        return pl.DataFrame()

    sort_cols = [effective_date_col, available_at_col]
    if revision_col in frame.columns:
        sort_cols.append(revision_col)
    frame = frame.sort(sort_cols)
    latest = frame.group_by(effective_date_col).tail(1).sort(effective_date_col)
    return latest.drop(available_at_col)


def hash_dataframe(df: pl.DataFrame) -> str:
    """Return a stable content hash for a Polars dataframe."""
    normalized = df.clone()
    date_col = DATE_COL if DATE_COL in normalized.columns else None
    if date_col is not None:
        normalized = normalized.with_columns(
            _normalized_daily_expr(normalized, date_col).alias(date_col)
        )
    payload = {
        "columns": [str(c) for c in normalized.columns],
        "dtypes": [str(normalized[c].dtype) for c in normalized.columns],
        "records": normalized.to_dicts(),
    }
    if date_col is not None:
        payload["index"] = [
            str(d)[:10] for d in normalized[date_col].to_list()
        ]
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
