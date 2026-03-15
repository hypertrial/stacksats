"""Weight API internals for model_development."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

DATE_COL = "date"


def _to_dt(val: dt.datetime | str) -> dt.datetime:
    if isinstance(val, dt.datetime):
        return val.replace(hour=0, minute=0, second=0, microsecond=0)
    return dt.datetime.strptime(str(val)[:10], "%Y-%m-%d")


def compute_weights_fast(
    features_df: pl.DataFrame,
    start_date: dt.datetime | str,
    end_date: dt.datetime | str,
    n_past: int | None = None,
    locked_weights: np.ndarray | None = None,
    *,
    compute_preference_scores_fn,
    allocate_sequential_stable_fn,
) -> pl.DataFrame:
    """Compute weights for a date window using precomputed features."""
    start_ts = _to_dt(start_date)
    end_ts = _to_dt(end_date)
    df = features_df.filter(
        (pl.col(DATE_COL) >= start_ts) & (pl.col(DATE_COL) <= end_ts)
    )
    if df.is_empty():
        return pl.DataFrame(schema={DATE_COL: pl.Datetime("us"), "weight": pl.Float64})

    n = len(df)
    preference_df = compute_preference_scores_fn(features_df, start_date, end_date)
    pref_col = "preference" if "preference" in preference_df.columns else preference_df.columns[-1]
    merged = df.select(DATE_COL).join(
        preference_df.select([DATE_COL, pl.col(pref_col).alias("_pref")]),
        on=DATE_COL,
        how="left",
    )
    pref_arr = merged["_pref"].fill_null(0.0).to_numpy()
    raw = (np.ones(n, dtype=float) / n) * np.exp(np.clip(pref_arr, -50, 50))

    if n_past is None:
        n_past = n
    weights = allocate_sequential_stable_fn(raw, n_past, locked_weights)

    dates = df[DATE_COL].to_list()
    return pl.DataFrame({DATE_COL: dates, "weight": weights.astype(float)})


def compute_window_weights(
    features_df: pl.DataFrame,
    start_date: dt.datetime | str,
    end_date: dt.datetime | str,
    current_date: dt.datetime | str,
    locked_weights: np.ndarray | None = None,
    *,
    validate_span_length_fn,
    compute_preference_scores_fn,
    compute_weights_from_target_profile_fn,
    assert_final_invariants_fn,
) -> pl.DataFrame:
    """Compute weights for a date range with lock-on-compute stability."""
    validate_span_length_fn(start_date, end_date)
    preference = compute_preference_scores_fn(
        features_df=features_df,
        start_date=start_date,
        end_date=end_date,
    )
    pref_col = "preference" if "preference" in preference.columns else "value"
    if pref_col not in preference.columns:
        raise ValueError("preference score frame must have 'date' and 'preference' columns.")
    preference = preference.select(DATE_COL, pl.col(pref_col).alias("value"))
    weights = compute_weights_from_target_profile_fn(
        features_df=features_df,
        start_date=start_date,
        end_date=end_date,
        current_date=current_date,
        target_profile=preference,
        mode="preference",
        locked_weights=locked_weights,
    )
    assert_final_invariants_fn(weights["weight"].to_numpy().astype(float))
    return weights
