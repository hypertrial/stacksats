"""Helpers for causal feature materialization and stable snapshot hashing."""

from __future__ import annotations

import hashlib
import json

import pandas as pd


VERSIONED_FEATURE_REQUIRED_COLUMNS = (
    "effective_date",
    "available_at",
)


def normalize_timestamp(value: pd.Timestamp | str) -> pd.Timestamp:
    """Normalize date-like values into timezone-naive daily timestamps."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def build_observed_frame(
    features_df: pd.DataFrame,
    *,
    start_date: pd.Timestamp | str,
    current_date: pd.Timestamp | str,
) -> pd.DataFrame:
    """Return a sorted copy restricted to the observed prefix."""
    if features_df.empty:
        observed_index = pd.date_range(
            normalize_timestamp(start_date),
            normalize_timestamp(current_date),
            freq="D",
        )
        return pd.DataFrame(index=observed_index)

    start_ts = normalize_timestamp(start_date)
    current_ts = normalize_timestamp(current_date)
    if current_ts < start_ts:
        return pd.DataFrame(
            columns=list(features_df.columns),
            index=pd.DatetimeIndex([], dtype="datetime64[ns]"),
        )

    frame = features_df.copy(deep=True)
    frame.index = pd.DatetimeIndex(frame.index).normalize()
    frame = frame.loc[~frame.index.duplicated(keep="last")].sort_index()
    return frame.loc[start_ts:current_ts].copy()


def materialize_versioned_observations(
    observations: pd.DataFrame,
    *,
    as_of_date: pd.Timestamp | str,
    effective_date_col: str = "effective_date",
    available_at_col: str = "available_at",
    revision_col: str = "revision_id",
) -> pd.DataFrame:
    """Select the latest available revision for each effective date."""
    if observations.empty:
        return pd.DataFrame()

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
    frame = observations.copy(deep=True)
    frame[effective_date_col] = pd.to_datetime(frame[effective_date_col]).dt.normalize()
    frame[available_at_col] = pd.to_datetime(frame[available_at_col]).dt.normalize()
    frame = frame.loc[frame[available_at_col] <= as_of_ts].copy()
    if frame.empty:
        return pd.DataFrame()

    sort_columns = [effective_date_col, available_at_col]
    ascending = [True, True]
    if revision_col in frame.columns:
        sort_columns.append(revision_col)
        ascending.append(True)
    frame = frame.sort_values(sort_columns, ascending=ascending)
    latest = frame.groupby(effective_date_col, sort=True, as_index=False).tail(1)
    latest = latest.set_index(effective_date_col).sort_index()
    return latest.drop(columns=[available_at_col], errors="ignore")


def hash_dataframe(df: pd.DataFrame) -> str:
    """Return a stable content hash for a dataframe."""
    normalized = df.copy(deep=True)
    normalized.index = pd.DatetimeIndex(normalized.index).normalize()
    payload = {
        "index": [ts.strftime("%Y-%m-%d") for ts in normalized.index],
        "columns": [str(column) for column in normalized.columns],
        "dtypes": [str(dtype) for dtype in normalized.dtypes],
        "records": normalized.where(pd.notna(normalized), None).to_dict(orient="records"),
    }
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
